import os
import gzip
import pandas as pd

def decompress_and_convert(input_dir, output_dir):
    """
    Decompress .tsv.gz files and convert them to .csv format.
    :param input_dir: Directory containing .tsv.gz files.
    :param output_dir: Directory to save the converted .csv files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all .tsv.gz files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tsv.gz"):
            print(f"Processing {file_name}...")

            # Define input and output file paths
            tsv_gz_path = os.path.join(input_dir, file_name)
            csv_file_name = file_name.replace(".tsv.gz", ".csv")
            csv_path = os.path.join(output_dir, csv_file_name)

            # Decompress and convert
            with gzip.open(tsv_gz_path, "rt") as tsv_file:
                df = pd.read_csv(tsv_file, sep="\t")  # Read TSV
                df.to_csv(csv_path, index=False)      # Save as CSV

            print(f"Saved {csv_file_name} to {output_dir}")

if __name__ == "__main__":
    input_dir = "./datasets"        # Directory containing .tsv.gz files
    output_dir = "./processed_datasets"  # Directory to save .csv files

    # Run the preprocessing function
    decompress_and_convert(input_dir, output_dir)