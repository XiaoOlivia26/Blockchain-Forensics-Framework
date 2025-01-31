import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "processed_datasets")
DYNAMIC_PARAMS_FILE = os.path.join(DATASETS_DIR, "dynamic_parameters.csv")

# Ensure output directory exists
os.makedirs(DATASETS_DIR, exist_ok=True)

# Function to preprocess datasets and calculate necessary parameters
def calculate_dynamic_parameters(input_dir, output_file):
    parameter_results = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv") and file_name != "dynamic_parameters.csv":
            print(f"Processing {file_name} for dynamic parameter calculation...")
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            if 'output_total_usd' not in df.columns or 'time' not in df.columns or 'block_id' not in df.columns:
                print(f"Skipping {file_name}: Required columns not found.")
                continue

            # Convert time to datetime format
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)

            # Lambda calculation
            theta = 0.5  # Decay to 50% over 24 hours
            delta_t = 24  # Total hours in a day
            lambda_value = -np.log(theta) / delta_t

            # Calculate recency in hours
            reference_time = df['time'].max()
            df['recency_hours'] = (reference_time - df['time']) / pd.Timedelta(hours=1)

            # Calculate weights
            df['weights'] = np.exp(-lambda_value * df['recency_hours'])

            # Calculate Weighted Mean and Weighted Std Dev
            weighted_mean = np.average(df['output_total_usd'], weights=df['weights'])
            weighted_std = np.sqrt(
                np.average(
                    (df['output_total_usd'] - weighted_mean) ** 2, 
                    weights=df['weights']
                )
            )

            # Calculate Alpha
            alpha = 0.5 * df['output_total_usd'].median() + 0.5 * weighted_mean

            # Calculate K with minimum value set to 2
            mad = np.median(np.abs(df['output_total_usd'] - df['output_total_usd'].median()))
            k = max(mad / weighted_mean, 0.01)  # Minimum K set to 0.01

            # Calculate Frequency Adjustment Term (F_i)
            df['block_frequency'] = df.groupby('block_id')['block_id'].transform('count')
            F_i = df['block_frequency'].mean()

            # Calculate Beta
            beta = 1 / (1 + np.log1p(df['block_frequency'].std())) if df['block_frequency'].std() > 0 else 0

            # Calculate Threshold
            threshold = weighted_mean + k * weighted_std + alpha / (1 + np.exp(-beta * F_i))

            # Store results
            parameter_results.append({
                "Dataset": file_name,
                "Lambda": round(lambda_value, 5),
                "Alpha": round(alpha, 2),
                "Beta": round(beta, 5),
                "K": round(k, 5),
                "Weighted Mean": round(weighted_mean, 2),
                "Weighted Std Dev": round(weighted_std, 2),
                "Threshold": round(threshold, 2),
                "Mean Block Frequency (F_i)": round(F_i, 2)
            })

    # Save to dynamic_parameters.csv
    result_df = pd.DataFrame(parameter_results)
    result_df = result_df[[
        "Dataset", "Lambda", "Alpha", "Beta", "K", 
        "Weighted Mean", "Weighted Std Dev", "Threshold", "Mean Block Frequency (F_i)"
    ]]
    result_df.to_csv(output_file, index=False)
    print("\nCalculated Dynamic Parameters:")
    print(result_df.to_string(index=False))
    print(f"\nDynamic parameters saved to {output_file}")

# Main execution
if __name__ == "__main__":
    print("Starting Dynamic Parameter Calculation with Minimum K=2 Adjustment...")
    calculate_dynamic_parameters(DATASETS_DIR, DYNAMIC_PARAMS_FILE)
    print("Dynamic Parameter Calculation complete.")
