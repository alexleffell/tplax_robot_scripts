import pandas as pd
import numpy as np
import argparse
import sys
import os

def main():
    """
    Main function to load, process, and merge data from two CSV files.
    """
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="""
        Merges two CSV files with different time bases.
        This script takes angle data from the first file, interpolates it to match the
        time base of the second file, renames the columns to '_angleEnc', and combines
        it with all the data from the second file.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--file1',
        type=str,
        required=True,
        help='Path to the first CSV file (containing node_angle/encoder data).'
    )
    parser.add_argument(
        '--file2',
        type=str,
        required=True,
        help='Path to the second CSV file (containing body_angle, position data, etc.).'
    )
    parser.add_argument(
        '--output',
        type=str,
        # NOTE: This argument is optional.
        help='Optional: Path for the output file. Defaults to the --file2 path with "_merged" appended.'
    )

    args = parser.parse_args()

    # --- 2. Determine Output Path ---
    if args.output is None:
        # Generate default output filename if not provided
        file2_root, file2_ext = os.path.splitext(args.file2)
        output_path = f"{file2_root}_merged{file2_ext}"
    else:
        output_path = args.output
        
    print(f"Input file 1: {args.file1}")
    print(f"Input file 2: {args.file2}")
    print(f"Output will be saved to: {output_path}")


    # --- 3. Load Data ---
    try:
        print("\nLoading data...")
        df1 = pd.read_csv(args.file1)
        df2 = pd.read_csv(args.file2)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file paths are correct.", file=sys.stderr)
        sys.exit(1)

    # --- 4. Data Preparation ---
    # Set time as the index for both dataframes
    df1.set_index('time', inplace=True)
    df2.set_index('time', inplace=True)

    # Select only the angle columns from the first file
    angle_columns_from_df1 = [col for col in df1.columns if 'angle' in col]
    df1_angles = df1[angle_columns_from_df1]

    # --- 5. Interpolation and Renaming ---
    # Reindex df1_angles to match the time index of df2 and interpolate
    print("Interpolating data...")
    df1_interpolated = df1_angles.reindex(df2.index).interpolate(method='linear')

    # Back-fill and forward-fill to handle any remaining NaNs at the edges
    df1_interpolated.bfill(inplace=True)
    df1_interpolated.ffill(inplace=True)

    # Rename the columns to the new '_angleEnc' format
    # e.g., 'node0_angle' becomes '0_angleEnc'
    new_column_names = {
        col: col.replace('node', '').replace('_angle', '') + '_angleEnc'
        for col in df1_interpolated.columns
    }
    df1_renamed = df1_interpolated.rename(columns=new_column_names)

    # --- 6. Merging ---
    # Combine the original second dataframe with the new, interpolated & renamed angle data
    print("Merging dataframes...")
    merged_df = pd.concat([df2, df1_renamed], axis=1)

    # --- 7. Save Output ---
    try:
        merged_df.to_csv(output_path)
        print(f"\n✅ Successfully merged the data and saved it to '{output_path}'!")
    except Exception as e:
        print(f"Error saving file to '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()