# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import ffmpeg
import argparse
from pupil_apriltags import Detector
import os
import re
import glob
import pandas as pd
import gc
import ast
import time
from scipy.signal import find_peaks
from pathlib import Path
start_time = time.time()

parser = argparse.ArgumentParser(description="AprilTag Tracker")
parser.add_argument("video_path", type=str, help="Path to the input video file")
parser.add_argument("--n_nodes", type=int, default=7, help="Number of nodes (default: 7)")
parser.add_argument("--nthreads", type=int, default=4, help="Number of threads (default: 4)")
parser.add_argument("--output_video", type=bool, default=False, help="Generate tagged video?")
parser.add_argument("--quad_dec", type=int, default=1, help="Generate tagged video?")

args = parser.parse_args()

dist_coeffs = np.array([-3.77932235e-01,  1.81918585e-01, -9.30045382e-05, 
                        -2.08242888e-03,  -4.90966330e-02])
camera_matrix = np.array([[1.54009298e+03, 0.00000000e+00, 9.58442344e+02],
                         [0.00000000e+00, 1.54227003e+03, 1.00722240e+03],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
tag_size = 0.045 # 0.025
tag_size_corner = 0.037
valid_tags = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,26,27,28,29]
video_path = args.video_path
qd = args.quad_dec
core_path = video_path[0:-4] # remove filetype

if args.output_video:
    output_path = core_path + "_tagged.mp4"
else:
    output_path = None

n_nodes = args.n_nodes
nthreads = args.nthreads


def interpolate_tracks(df, max_timesteps=1000):
    """
    Interpolate missing timesteps in particle tracks, maintaining existing data
    and only filling gaps with increments of 1.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing particle tracks with columns 
                      ['node_id', 'frame#', 'x', 'y', 'z', 'angle']
    max_timesteps (int): Maximum number of timesteps expected (0 to max_timesteps-1)
    
    Returns:
    pd.DataFrame: DataFrame with interpolated tracks
    """
    # Create a list to store interpolated data for each particle
    interpolated_data = []
    
    # Get unique particle IDs
    particle_ids = df['node_id'].unique()
    
    # Process each particle
    for particle_id in particle_ids:
        # Get data for this particle
        particle_data = df[df['node_id'] == particle_id].copy()
        
        # Create a DataFrame with all timesteps from 0 to max_timesteps-1
        all_timesteps = pd.DataFrame({
            'frame#': range(max_timesteps),
            'node_id': particle_id
        })
        
        # Merge with actual data
        merged = pd.merge(all_timesteps, particle_data, 
                         on=['node_id', 'frame#'], 
                         how='left')
        
        # Interpolate missing values
        merged['x'] = merged['x'].interpolate(method='linear', limit_direction='both')
        merged['y'] = merged['y'].interpolate(method='linear', limit_direction='both')
        merged['z'] = merged['z'].interpolate(method='linear', limit_direction='both')
        merged['angle'] = merged['angle'].interpolate(method='linear', limit_direction='both')
        
        interpolated_data.append(merged)
    
    # Combine all interpolated tracks
    interpolated_df = pd.concat(interpolated_data, ignore_index=True)
    
    # Sort by particle_id and timestamp
    interpolated_df = interpolated_df.sort_values(['node_id', 'frame#'])
    
    # Calculate statistics about interpolation
    stats = calculate_interpolation_stats(df, interpolated_df)
    print_interpolation_stats(stats)
    
    return interpolated_df

def calculate_interpolation_stats(original_df, interpolated_df):
    """
    Calculate statistics about the interpolation process.
    """
    stats = {
        'original_rows': len(original_df),
        'interpolated_rows': len(interpolated_df),
        'particles': len(original_df['node_id'].unique()),
        'added_points': len(interpolated_df) - len(original_df)
    }
    
    # Calculate gaps filled per particle
    particle_stats = []
    gap_sizes = []
    for particle_id in original_df['node_id'].unique():
        # Original timestamps for this particle
        orig_times = set(original_df[original_df['node_id'] == particle_id]['frame#'])
        
        # Get all timestamps after interpolation
        interp_times = set(interpolated_df[interpolated_df['node_id'] == particle_id]['frame#'])
        
        # Calculate gaps
        filled_points = len(interp_times - orig_times)
        total_points = len(interp_times)
        
        # Calculate gap sizes
        orig_times_list = sorted(list(orig_times))
        for i in range(len(orig_times_list) - 1):
            gap = orig_times_list[i + 1] - orig_times_list[i] - 1
            if gap > 0:
                gap_sizes.append(gap)
        
        particle_stats.append((filled_points / total_points) * 100)
    
    stats['avg_interpolated_percentage'] = np.mean(particle_stats)
    stats['max_interpolated_percentage'] = np.max(particle_stats)
    stats['avg_gap_size'] = np.mean(gap_sizes) if gap_sizes else 0
    stats['max_gap_size'] = np.max(gap_sizes) if gap_sizes else 0
    print(original_df['node_id'].unique())
    
    return stats

def print_interpolation_stats(stats):
    """
    Print statistics about the interpolation process.
    """
    print("\nInterpolation Statistics:")
    print(f"Original number of points: {stats['original_rows']}")
    print(f"Interpolated number of points: {stats['interpolated_rows']}")
    print(f"Number of particles: {stats['particles']}")
    print(f"Added points: {stats['added_points']}")
    print(f"Average percentage of interpolated points per particle: {stats['avg_interpolated_percentage']:.2f}%")
    print(f"Maximum percentage of interpolated points for any particle: {stats['max_interpolated_percentage']:.2f}%")
    print(f"Average gap size: {stats['avg_gap_size']:.2f}")
    print(f"Maximum gap size: {stats['max_gap_size']}")

def generate_dataframe(df, total_frames, fps, n_nodes=7, dual=False):
    """
    takes in a dataframe output by the tag tracker and reformats it to be more convenient. 
    Also:
    - interpolates missing node locations
    - calculates body angle and adds the in-frame castor rotation 
    - 
    file - string of file location of csv file of tag tracker output
    """
    df_int = interpolate_tracks(df,total_frames)
    
    corner_ids = [26,27,28,29]
    robot_list = []
    corners = [[],[],[],[]]
    corner_locs = []
    if dual:
        for t in range(total_frames):
            temp_df = df_int[df_int["frame#"]==t]
            row = {'time':t/fps}
            # Get position and
            centroid_x = 0
            centroid_y = 0
            for n in range(n_nodes):
                try:
                    node0 = temp_df[temp_df["node_id"]==n]
                    node1 = temp_df[temp_df["node_id"]==n+n_nodes]
                    row[f'{n}_x'] = (node0['x'].iloc[0] + node1['x'].iloc[0])/2
                    row[f'{n}_y'] = (node0['y'].iloc[0] + node1['y'].iloc[0])/2
                    row[f'{n}_z'] = (node0['z'].iloc[0] + node1['z'].iloc[0])/2
                    node_dx = node0['x'].iloc[0] - node1['x'].iloc[0]
                    node_dy = node0['y'].iloc[0] - node1['y'].iloc[0]
                    row[f'{n}_raw_angle'] = np.arctan2(node_dy,node_dx)
                    centroid_x += row[f'{n}_x']/n_nodes
                    centroid_y += row[f'{n}_y']/n_nodes
                except:
                    pass
            try:
                dx = row['0_x']-row['1_x']
                dy = row['0_y']-row['1_y']
                theta = np.arctan2(dy,dx)
                row['body_angle'] = theta
                row['centroid_x'] = centroid_x
                row['centroid_y'] = centroid_y
            except:
                pass
            
            for n in range(n_nodes):
                try:
                    row[f'{n}_angle'] = row[f'{n}_raw_angle'] - theta
                except:
                    pass
            robot_list.append(row)
            # get corner locations
            for n in range(len(corner_ids)):
                try:
                    node = df_int[(df_int["frame#"]==t) & (df_int["node_id"]==corner_ids[n])]
                    corners[n].append( (node['x'].iloc[0],node['y'].iloc[0]) )
                except:
                    pass
             
        for n in range(len(corner_ids)):
            corner_locs.append( tuple(float(np.mean(values)) for values in zip(*corners[n])))
            
    else:
        for t in range(total_frames):
            center = df_int[(df_int["frame#"]==t) & (df_int["node_id"]==0)]
            one = df_int[(df_int["frame#"]==t) & (df_int["node_id"]==1)]
            dx = center["x"].iloc[0] - one["x"].iloc[0]
            dy = center["y"].iloc[0] - one["y"].iloc[0]
            theta = np.arctan2(dy,dx)
            row = {'time':(t/fps), 'body_angle':theta}
            centroid_x = 0
            centroid_y = 0
            for n in range(n_nodes):
                try:
                    node = df_int[(df_int["frame#"]==t) & (df_int["node_id"]==n)]
                    row[str(n)+'_x'] = node['x'].iloc[0]
                    row[str(n)+'_y'] = node['y'].iloc[0]
                    row[str(n)+'_z'] = node['z'].iloc[0]
                    row[str(n)+'_raw_angle'] = node['angle'].iloc[0]
                    row[str(n)+'_angle'] = node['angle'].iloc[0] - theta
                    centroid_x += row[f'{n}_x']/n_nodes
                    centroid_y += row[f'{n}_y']/n_nodes
                except:
                    print(f"Failed to add node {n} at frame {t}")
                    pass
            try:
                dx = row['0_x']-row['1_x']
                dy = row['0_y']-row['1_y']
                theta = np.arctan2(dy,dx)
                row['body_angle'] = theta
                row['centroid_x'] = centroid_x
                row['centroid_y'] = centroid_y
            except:
                pass
            # get corner locations
            for n in range(len(corner_ids)):
                try:
                    node = df_int[(df_int["frame#"]==t) & (df_int["node_id"]==corner_ids[n])]
                    corners[n].append( (node['x'].iloc[0],node['y'].iloc[0]) )
                except:
                    pass
                
            robot_list.append(row) 
        for n in range(len(corner_ids)):
            corner_locs.append( tuple(float(np.mean(values)) for values in zip(*corners[n])))

    robot_df = pd.DataFrame(robot_list)
    robot_df.attrs['corner_locs'] = corner_locs
    robot_df.attrs['n_nodes'] = n_nodes
    # Save DataFrame to CSV with attributes in the header
    with open(core_path+'_robot.csv', "w") as f:
        for key, value in robot_df.attrs.items():
            f.write(f"# {key}: {value}\n")  # Write metadata as comments
        robot_df.to_csv(f, index=False)
    return robot_df

at_detector = Detector(
                       families='tag16h5',
                       nthreads=nthreads,
                       quad_decimate=qd,
                       quad_sigma=0.,
                       refine_edges=1,
                       decode_sharpening=0.0,
                       debug=0)

# Ensure input video exists
if not os.path.exists(video_path):
    print(f"Error: Input video not found at {video_path}")
    

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {total_frames} frames, {frame_width}x{frame_height}, {fps} FPS")

# Create video writer if output path is specified
if output_path:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Writing output to: {os.path.abspath(output_path)}")

frame_count = 0
results = []
# Iterate over all frames
while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    # convert to gray and detect tags    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect markers
    tags = at_detector.detect(image)
    
    for r in tags:
        if r.hamming < 1 and r.tag_id in valid_tags and r.decision_margin > 1.:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = tuple(map(int, ptA))
            ptB = tuple(map(int, ptB))
            ptC = tuple(map(int, ptC))
            ptD = tuple(map(int, ptD))

            # Draw bounding box
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)

            # Draw the center
            center = tuple(map(int, r.center))
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            cv2.putText(image, f"ID: {r.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Pose estimation (distance calculation)
            if r.tag_id in valid_tags[0:-4]:
                obj_points = np.array([
                    [-tag_size / 2,  tag_size / 2, 0],   # Top-left corner
                    [ tag_size / 2,  tag_size / 2, 0],  # Top-right corner
                    [ tag_size / 2, -tag_size / 2, 0],  # Bottom-right corner
                    [-tag_size / 2, -tag_size / 2, 0]  # Bottom-left corner
                ], dtype=np.float32)
            else:
                obj_points = np.array([
                    [-tag_size_corner / 2,  tag_size_corner / 2, 0],   # Top-left corner
                    [ tag_size_corner / 2,  tag_size_corner / 2, 0],  # Top-right corner
                    [ tag_size_corner / 2, -tag_size_corner / 2, 0],  # Bottom-right corner
                    [-tag_size_corner / 2, -tag_size_corner / 2, 0]  # Bottom-left corner
                ], dtype=np.float32)                

            img_points = np.array(r.corners, dtype=np.float32)
                
            # Solve IPPE PnP (Perspective-n-Point) to find pose
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs,flags = cv2.SOLVEPNP_IPPE_SQUARE)

            if success: 
                # tvec contains the translation vector, where tvec[2] is the distance
                distance = tvec[2][0]
                
                results.append({
                    'frame#': frame_count,
                    'node_id': r.tag_id,
                    'x': tvec[0][0],
                    'y': tvec[1][0],
                    'z': tvec[2][0],
                    'angle': np.arctan2(rvec[1], rvec[0])[0]
                })


    # Write frame if output path is specified
    if output_path:
        # Undistort the frame
        h, w = image.shape[:2]  # Frame dimensions
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        flip = cv2.flip(undistorted_image, 0)
        out.write(flip)

    # Display current frame number
    frame_count += 1
    if frame_count % 100 == 0:  # Periodic cleanup
        print(f"Processed {frame_count}/{total_frames} frames")
#         gc.collect()    


# Release resources
cap.release()
if output_path:
    out.release()
    print(f"Output video saved to: {os.path.abspath(output_path)}")
    
df = pd.DataFrame(results)
df.to_csv(core_path + "_raw.csv", index=False)
generate_dataframe(df, total_frames, fps, n_nodes=n_nodes, dual=False)
print("Processing complete.")
print("--- %s seconds ---" % (time.time() - start_time))
