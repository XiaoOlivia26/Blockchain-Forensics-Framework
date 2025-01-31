import os
import requests

# Define the base URL for blockchain transaction dumps
base_url = "https://gz.blockchair.com"

# Define the directories
current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(current_dir, "datasets")

# Ensure the datasets directory exists
os.makedirs(datasets_dir, exist_ok=True)

def download_dataset(date, blockchain):
    """
    Download dataset for the given date and blockchain.
    """
    file_name = f"blockchair_{blockchain}_transactions_{date.replace('-', '')}.tsv.gz"
    url = f"{base_url}/{blockchain}/transactions/{file_name}"
    save_path = os.path.join(datasets_dir, file_name)

    try:
        print(f"Downloading {file_name}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved to {save_path}")
        else:
            print(f"File not found: {url}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    date = input("Enter the date of the dataset you want to download (YYYY-MM-DD): ")
    blockchains = ["dogecoin", "bitcoin"]
    for blockchain in blockchains:
        download_dataset(date, blockchain)

if __name__ == "__main__":
    main()

