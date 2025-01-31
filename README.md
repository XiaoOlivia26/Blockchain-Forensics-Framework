# Blockchain-Forensics-Framework

This project provides a framework for analyzing cryptocurrency transaction datasets, specifically Bitcoin and Dogecoin, to detect arbitrage opportunities, anomalous transactions, and high-risk blocks. The framework includes modules for downloading, preprocessing, analyzing, and visualizing the datasets.

---

## Features

- **Dataset Selection and Downloading**: Select a specific date to download Bitcoin and Dogecoin datasets.
- **Data Preprocessing**: Convert raw datasets from `.tsv.gz` format to `.csv` and clean the data for analysis.
- **Anomaly Detection**: Use Isolation Forest to detect anomalous transactions.
- **Arbitrage Opportunity Detection**: Identify cross-currency arbitrage opportunities between Bitcoin and Dogecoin.
- **High-Risk Block Detection**: Identify high-risk blocks based on transaction count and total value.
- **Visualization**: Generate visualizations of high-risk transactions, blocks, arbitrage opportunities, and correlation heatmaps.
- **Modular Workflow**: The framework is designed to be modular, enabling easy integration into other projects.

---

## Repository Structure

```yaml
.
├── datasets/                           # Raw downloaded datasets
├── processed_datasets/                 # Preprocessed datasets in .csv format
├── results/                            # Analysis results (e.g., filtered arbitrage, anomalies)
├── figures/                            # Generated visualizations (e.g., clusters, heatmaps)
├── download_datasets.py                # Script to download datasets for a specific date
├── preprocess_data.py                  # Script to preprocess and convert datasets
├── data_processing.py                  # Script for calculating dynamic parameters and threshold
├── blockchain_forensics_workflow.py    # Main script that integrates the workflow
└── README.md                           # Project documentation
```

## Scripts Overview
1. **`download_datasets.py`**  
   - Allows users to select a specific date for downloading Bitcoin and Dogecoin datasets from Blockchair.
   - Outputs: Compressed raw datasets in `datasets/`.

2. **`preprocess_data.py`**  
   - Decompresses `.tsv.gz` files and converts them to `.csv` format for analysis.
   - Outputs: Preprocessed `.csv` files in `processed_datasets/` directory.

3. **`data_processing.py`**  
   - Calculates dynamic parameters such as the weighted mean, standard deviation, and thresholds for anomaly detection.
   - Outputs: `dynamic_parameters.csv` in the `processed_datasets/` directory.

4. **`blockchain_forensics_workflow.py`**  
   - Integrates the entire workflow including anomaly detection, and arbitrage analysis.
   - Outputs: Results and visualizations, including high-risk transactions, blocks, and arbitrage opportunities in the `results/` and `figures/`.

---

## Generated Figures
The following figures are generated throughout the workflow:
- **Figure 1**: High-risk block analysis for Bitcoin.
- **Figure 2**: High-risk block analysis for Dogecoin.
- **Figure 3**: High-risk transactions analysis for Bitcoin.
- **Figure 4**: High-risk transactions analysis for Dogecoin.
- **Figure 5**: Anomalous transactions
- **Figure 6**: Arbitrage opportunities
---

## Setup Instructions

```
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crypto-arbitrage-framework.git
cd crypto-arbitrage-framework
```

### 2.  Install Dependencies
You can install the necessary dependencies as follows (they are included in the script `blockchain_forensics_workflow.py`):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### 3. Run the Scripts
Follow these steps to execute the workflow:
#### Download the datasets: 
Run download_datasets.py and specify the date of the datasets you want to download:
```bash
python download_datasets.py
```

Example interaction:
```css
Enter the date of the dataset you want to download (YYYY-MM-DD): 2024-11-22
```
The dataset will be saved to the datasets/ directory.

#### Preprocess the datasets:
Unzip the raw datasets for further analysis:
```bash
python preprocess_data.py
```

#### Process dynamic parameters and thresholds:
Run the data processing script to handle threshold and dynamic parameter calculations:
```bash
python data_processing.py
```

#### Execute the full analysis pipeline:
Run the main workflow to perform analysis and generate results:
```bash
python blockchain_forensics_workflow.py
```

## Notes
- The framework is modular, and each script can be run independently.
- Ensure the datasets/ folder has sufficient disk space for downloading large files.
- If the datasets for a specific date are unavailable, you will receive an error message during the download process.

## License
[MIT](https://choosealicense.com/licenses/mit/)

