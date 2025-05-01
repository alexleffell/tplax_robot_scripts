import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Interpolate multi-node sensor data to a common time base.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-f", "--frequency", type=float, default=None, help="Sampling frequency in Hz")
    parser.add_argument("-o", "--output_data", default="interpolated_data.csv", help="Path to save interpolated CSV")
    parser.add_argument("-p", "--output_plot", default="interpolation_plot.png", help="Path to save comparison plot")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df['timestamp'] = df['timestamp_s'] + df['timestamp_us'] * 1e-6
    df = df.drop(columns=['timestamp_s', 'timestamp_us'])

    # Estimate sampling frequency if not provided
    if args.frequency is None:
        est_freqs = []
        for node_id in df['node_id'].unique():
            t = df[df['node_id'] == node_id].sort_values('timestamp')['timestamp'].values
            dt = np.diff(t)
            est_freqs.append(1.0 / np.mean(dt))
        sample_frequency = np.mean(est_freqs)
        print(f"[Info] Estimated mean sample frequency: {sample_frequency:.2f} Hz")
    else:
        sample_frequency = args.frequency
        print(f"[Info] Using user-provided sample frequency: {sample_frequency:.2f} Hz")

    sample_period = 1.0 / sample_frequency

    # Define common timebase window from overlapping regions
    start_time = df.groupby('node_id')['timestamp'].min().max()
    end_time = df.groupby('node_id')['timestamp'].max().min()

    if start_time >= end_time:
        raise ValueError("Timestamps of different nodes do not overlap.")

    num_samples = int((end_time - start_time) / sample_period) + 1
    common_time = np.linspace(0, (num_samples - 1) * sample_period, num=num_samples)
    common_time = np.round(common_time, 3)  # millisecond resolution

    output = {'time': common_time}
    stats = []
    node_ids = sorted(df['node_id'].unique())
    node_stats = {}

    for node_id in node_ids:
        node_data = df[df['node_id'] == node_id].sort_values('timestamp')
        t = node_data['timestamp'].values
        angle = node_data['angle_value'].values
        encoder = node_data['encoder_value'].values

        # Only include data within interpolation window
        mask = (t >= start_time) & (t <= end_time)
        t = t[mask]
        angle = angle[mask]
        encoder = encoder[mask]

        t_shifted = t - start_time

        # Interpolate to common time base
        angle_interp = interp1d(t_shifted, angle, kind='linear', bounds_error=False, fill_value='extrapolate')
        encoder_interp = interp1d(t_shifted, encoder, kind='linear', bounds_error=False, fill_value='extrapolate')

        output[f'node{node_id}_angle'] = angle_interp(common_time)
        output[f'node{node_id}_encoder'] = encoder_interp(common_time)

        # Calculate deviations once for both statistics and plotting
        deviations = []
        nearest_indices = []
        
        for timestamp in t_shifted:
            # Find index of closest common timebase point
            idx = np.abs(common_time - timestamp).argmin()
            nearest_indices.append(idx)
            nearest_common_time = common_time[idx]
            deviation = timestamp - nearest_common_time
            deviations.append(deviation)
        
        deviations = np.array(deviations)
        
        # Store all data for this node
        node_stats[node_id] = {
            'timestamps': t_shifted,
            'angles': angle,
            'encoders': encoder,
            'deviations': deviations,
            'deviations_ms': deviations * 1000,  # Convert to ms for plotting
            'nearest_indices': nearest_indices
        }
        
        # Add to stats DataFrame
        stats.append({
            'node_id': node_id,
            'num_samples': len(t),
            'mean_deviation_ms': np.mean(np.abs(deviations)) * 1000,
            'max_deviation_ms': np.max(np.abs(deviations)) * 1000,
            'min_deviation_ms': np.min(np.abs(deviations)) * 1000,
            'std_deviation_ms': np.std(deviations) * 1000
        })

    # Save interpolated data
    aligned_df = pd.DataFrame(output)
    aligned_df.to_csv(args.output_data, index=False)
    print(f"[Saved] Interpolated data -> {os.path.abspath(args.output_data)}")

    # Print stats
    stats_df = pd.DataFrame(stats)
    print("\nImproved Statistics (deviation from common timebase):")
    print(stats_df.to_string(index=False))

    # Plot angle comparison
    fig, axes = plt.subplots(len(node_ids), 1, figsize=(10, 3 * len(node_ids)), sharex=True)
    if len(node_ids) == 1:
        axes = [axes]

    # Add a plot showing deviations
    fig2, axes2 = plt.subplots(len(node_ids), 1, figsize=(10, 3 * len(node_ids)), sharex=True)
    if len(node_ids) == 1:
        axes2 = [axes2]

    # Now use the pre-calculated data for plotting
    for ax, ax2, node_id in zip(axes, axes2, node_ids):
        # Use the pre-calculated data
        shifted_time = node_stats[node_id]['timestamps']
        node_angles = node_stats[node_id]['angles']
        deviations_ms = node_stats[node_id]['deviations_ms']
        
        # Data visualization
        ax.plot(shifted_time, node_angles, 'o', label='Raw', markersize=3, alpha=0.6)
        ax.plot(common_time, aligned_df[f'node{node_id}_angle'], '-', label='Interpolated', linewidth=1)
        ax.set_title(f'Node {node_id} - Angle')
        ax.set_ylabel('Angle')
        ax.grid(True)
        ax.legend()
        
        # Plot deviations using pre-calculated values
        ax2.plot(shifted_time, deviations_ms, 'o-', markersize=2)
        ax2.set_title(f'Node {node_id} - Time Deviation from Common Timebase')
        ax2.set_ylabel('Deviation (ms)')
        ax2.grid(True)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes2[-1].set_xlabel('Time (s)')
    
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"[Saved] Comparison plot -> {os.path.abspath(args.output_plot)}")
    
    plt.figure(fig2.number)
    plt.tight_layout()
    deviation_plot = args.output_plot.replace('.png', '_deviations.png')
    plt.savefig(deviation_plot)
    print(f"[Saved] Deviation plot -> {os.path.abspath(deviation_plot)}")

if __name__ == "__main__":
    main()