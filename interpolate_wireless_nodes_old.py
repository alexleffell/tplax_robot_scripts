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

        # Stats
        expected_times = np.arange(t_shifted[0], t_shifted[-1], sample_period)
        actual_sample_times = np.interp(expected_times, t_shifted, t_shifted)
        jitter = actual_sample_times - expected_times
        drift = t_shifted[-1] - (t_shifted[0] + (len(t_shifted) - 1) * sample_period)

        stats.append({
            'node_id': node_id,
            'num_samples': len(t),
            'mean_jitter_ms': np.mean(np.abs(jitter)) * 1000,
            'max_jitter_ms': np.max(np.abs(jitter)) * 1000,
            'drift_ms': drift * 1000
        })

    # Save interpolated data
    aligned_df = pd.DataFrame(output)
    aligned_df.to_csv(args.output_data, index=False)
    print(f"[Saved] Interpolated data -> {os.path.abspath(args.output_data)}")

    # Print stats
    stats_df = pd.DataFrame(stats)
    print("\nInterpolation Statistics:")
    print(stats_df.to_string(index=False))

    # Plot angle comparison
    fig, axes = plt.subplots(len(node_ids), 1, figsize=(10, 3 * len(node_ids)), sharex=True)
    if len(node_ids) == 1:
        axes = [axes]

    for ax, node_id in zip(axes, node_ids):
        node_data = df[df['node_id'] == node_id].sort_values('timestamp')
        shifted_time = node_data['timestamp'] - start_time
        ax.plot(shifted_time, node_data['angle_value'], 'o', label='Raw', markersize=3, alpha=0.6)
        ax.plot(common_time, aligned_df[f'node{node_id}_angle'], '-', label='Interpolated', linewidth=1)
        ax.set_title(f'Node {node_id} - Angle')
        ax.set_ylabel('Angle')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"[Saved] Comparison plot -> {os.path.abspath(args.output_plot)}")

if __name__ == "__main__":
    main()
