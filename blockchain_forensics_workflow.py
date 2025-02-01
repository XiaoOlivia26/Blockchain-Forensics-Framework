import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.ensemble import IsolationForest
from matplotlib.gridspec import GridSpec
from matplotlib import dates as mdates

# First import seaborn
sns.set_theme(style="whitegrid")  
plt.style.use('default')  # Use default matplotlib style as base

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATASETS_DIR = os.path.join(BASE_DIR, "processed_datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_dataset_by_keyword(keyword):
    """Load dataset by keyword and extract date"""
    for file_name in os.listdir(PROCESSED_DATASETS_DIR):
        if keyword in file_name:
            file_path = os.path.join(PROCESSED_DATASETS_DIR, file_name)
            date = get_dataset_date(file_name)
            return pd.read_csv(file_path), date
    return None, None

def get_dataset_date(dataset_name):
    """Extract date from dataset name"""
    match = re.search(r'(\d{8})', dataset_name)
    return match.group(1) if match else None

def preprocess_data(data):
    """Preprocess the dataset"""
    data = data.copy()
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])
    return data

def load_threshold(currency_keyword):
    """Load threshold value for a specific currency from dynamic_parameters.csv"""
    # Path to the dynamic parameters file
    parameters_file_path = os.path.join(PROCESSED_DATASETS_DIR, "dynamic_parameters.csv")
    
    try:
        # Load the dynamic parameters file
        params_df = pd.read_csv(parameters_file_path)
        
        # Filter the row based on the currency keyword
        specific_row = params_df[params_df['Dataset'].str.contains(currency_keyword, case=False, regex=False)]
        
        if not specific_row.empty:
            # Return the threshold value from the filtered row
            return specific_row['Threshold'].values[0]
        else:
            print(f"No threshold found for {currency_keyword}, using default value 1000")
            return 1000  # Default threshold if no specific value found
    except FileNotFoundError:
        print(f"Parameter file not found at {parameters_file_path}, using default value 1000")
        return 1000  # Default threshold if file not found
    except Exception as e:
        print(f"Error loading threshold for {currency_keyword}: {str(e)}")
        return 1000  # Default threshold if any other error occurs

def filter_transactions(data, threshold):
    """Filter transactions based on threshold"""
    return data[data['output_total_usd'] > threshold].copy()

def cluster_by_block(data):
    """Perform block-level analysis"""
    block_stats = data.groupby('block_id').agg({
        'output_total_usd': ['sum', 'mean', 'std'],
        'fee_usd': ['sum', 'mean'],
        'time': ['min', 'max'],
        'hash': 'count'
    }).reset_index()
    
    block_stats.columns = ['block_id', 'total_value', 'mean_value', 'std_value',
                          'total_fee', 'mean_fee', 'start_time', 'end_time', 
                          'transaction_count']
    return block_stats

def detect_high_risk_blocks(block_stats):
    """Detect high-risk blocks"""
    return block_stats[
        (block_stats['transaction_count'] > block_stats['transaction_count'].quantile(0.85)) |
        (block_stats['total_value'] > block_stats['total_value'].quantile(0.85))
    ].copy()

def match_cross_currency_transactions(btc_data, doge_data, time_window=600):
    """Match transactions across currencies"""
    btc_data = btc_data.copy()
    doge_data = doge_data.copy()
    
    btc_data['time'] = pd.to_datetime(btc_data['time'])
    doge_data['time'] = pd.to_datetime(doge_data['time'])
    
    return pd.merge_asof(
        btc_data.sort_values('time'),
        doge_data.sort_values('time'),
        on='time',
        tolerance=pd.Timedelta(seconds=time_window),
        direction='nearest',
        suffixes=('_btc', '_doge')
    )

def visualize_high_risk_transactions(data, name, date, figure_path):
    """Enhanced visualization of high-risk transactions"""
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Time series heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    data['hour'] = data['time'].dt.hour
    hourly_stats = data.groupby('hour').agg({
        'output_total_usd': ['count', 'mean', 'sum']
    })
    
    sns.heatmap(hourly_stats, 
                ax=ax1, 
                cmap='YlOrRd',
                annot=True,
                fmt='.1f')
    ax1.set_title('Hourly Transaction Patterns', fontsize=12)
    
    # Value distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=data, y='output_total_usd', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Value Distribution', fontsize=12)
    
    # Risk metrics
    ax3 = fig.add_subplot(gs[1, 0])
    data['zscore'] = (data['output_total_usd'] - data['output_total_usd'].mean()) / data['output_total_usd'].std()
    sns.scatterplot(data=data,
                    x='output_total_usd',
                    y='fee_usd',
                    size='zscore',
                    sizes=(50, 400),
                    alpha=0.6,
                    ax=ax3)
    ax3.set_title('Risk Analysis', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Volume analysis
    ax4 = fig.add_subplot(gs[1, 1])
    hourly_volume = data.groupby('hour')['output_total_usd'].sum()
    ax4.bar(hourly_volume.index, hourly_volume.values, alpha=0.7)
    ax4.set_title('Hourly Volume', fontsize=12)
    
    plt.suptitle(f'{name.title()} High-Risk Analysis - {date}', fontsize=14)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_high_risk_blocks(blocks, name, date, figure_path):
    """Visualize high-risk blocks with consistent sizing and formatting"""
    fig = plt.figure(figsize=(20, 15))
    
    gs = GridSpec(2, 2, figure=fig, 
                 height_ratios=[1, 1.5], 
                 hspace=0.3, 
                 wspace=0.3)
    
    # Top plot - Metric comparison
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['total_value', 'transaction_count', 'total_fee']
    normalized_data = blocks[metrics].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Select top N blocks by total value
    top_blocks = blocks.nlargest(min(10, len(blocks)), 'total_value')
    norm_data = top_blocks[metrics].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Create heatmap
    sns.heatmap(norm_data.T, 
                ax=ax1,
                cmap='viridis',
                xticklabels=[f'Block {i+1}' for i in range(len(top_blocks))],
                yticklabels=metrics,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Normalized Value'})
    
    ax1.set_title(f'High-Risk {name.title()} Blocks Analysis', 
                  fontsize=14, pad=20)
    
    # Value formatter for large numbers
    def value_formatter(x, p):
        if x >= 1e9:
            return f'${x/1e9:.1f}B'
        elif x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.1f}K'
        else:
            return f'${x:.0f}'
    
    # Left bottom - Transaction scatter
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(blocks['transaction_count'], 
                         blocks['total_value'], 
                         c=blocks['total_fee'], 
                         cmap='viridis',
                         s=100)
    
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(value_formatter))
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Total Fee (USD)', fontsize=12)
    ax2.set_title('Transaction Count vs Total Value', fontsize=14)
    ax2.set_xlabel('Transaction Count', fontsize=12)
    ax2.set_ylabel('Total Value (USD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Right bottom - Fee distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=blocks, y='total_fee', ax=ax3)
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(value_formatter))
    ax3.set_title('Fee Distribution', fontsize=14)
    ax3.set_ylabel('Total Fee (USD)', fontsize=12)
    
    # Overall title
    plt.suptitle(f'Block Analysis for {name.title()} - {date}', 
                 fontsize=16, y=0.95)
    
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"High-risk blocks visualization saved to {figure_path}")

def visualize_anomalies(data, anomalous_transactions, title, figure_path):
    """Visualize anomalous transactions with bubble plot"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    def currency_formatter(x, p):
        if x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.1f}K'
        else:
            return f'${x:.0f}'
    
    # Plot only anomalous transactions
    scatter = ax.scatter(anomalous_transactions['time'], 
                        anomalous_transactions['output_total_usd_btc'],
                        c=anomalous_transactions['fee_usd_btc'],
                        s=100,
                        cmap='viridis',
                        alpha=0.7)
    
    # Format axes
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    ax.set_title(f'{title} - Time Series', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Transaction Value (USD)')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Transaction Fee (USD)')
    
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Anomaly visualization saved to {figure_path}")

def visualize_arbitrage_opportunities(data, title, figure_path):
    """Visualize arbitrage opportunities with clean design"""
    if data.empty:
        print("No profitable arbitrage opportunities to visualize")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot by direction with small dots
    colors = {'BTC->DOGE': 'blue', 'DOGE->BTC': 'red'}
    for direction in data['direction'].unique():
        direction_data = data[data['direction'] == direction]
        ax.scatter(direction_data['time'], 
                  direction_data['profit_percentage'],
                  c=colors[direction],
                  s=20,
                  alpha=0.6,
                  label=direction)
    
    # Format axes
    def percentage_formatter(x, p):
        return f'{x:.1f}%'
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(percentage_formatter))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Profit (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Direction', loc='upper right')
    
    # Set title matching anomalous_transactions style
    ax.set_title(f'{title} - Time Series', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Arbitrage visualization saved to {figure_path}")
    
def detect_anomalous_transactions(data):
    """Detect anomalous transactions using Isolation Forest"""
    try:
        # Create copy to avoid modifications to original data
        data = data.copy()
        
        # Select features for anomaly detection
        features = ['input_total_usd_btc', 'output_total_usd_btc', 'fee_usd_btc']
        
        # Print debug information
        print("Data shape:", data[features].shape)
        print("Features sample:", data[features].head())
        
        # Initialize and fit IsolationForest
        model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        # Detect anomalies
        data['anomaly'] = model.fit_predict(data[features])
        
        # Filter anomalous transactions (where prediction is -1)
        anomalous_transactions = data[data['anomaly'] == -1]
        
        print(f"Detected {len(anomalous_transactions)} anomalous transactions")
        return anomalous_transactions
        
    except Exception as e:
        print(f"Error in anomaly detection: {str(e)}")
        raise

def analyze_arbitrage_opportunities(btc_data, doge_data):
    """Analyze bidirectional arbitrage opportunities with detailed profit analysis"""
    # Prepare data
    btc_data, doge_data = btc_data.copy(), doge_data.copy()
    btc_data['time'] = pd.to_datetime(btc_data['time'])
    doge_data['time'] = pd.to_datetime(doge_data['time'])
    
    print(f"\nAnalyzing {len(btc_data)} BTC and {len(doge_data)} DOGE transactions")
    
    # Match transactions within time window
    matched_data = pd.merge_asof(
        btc_data.sort_values('time'),
        doge_data.sort_values('time'),
        on='time',
        tolerance=pd.Timedelta(seconds=600),
        suffixes=('_btc', '_doge')
    )
    
    if matched_data.empty:
        print("No matching transactions found")
        return pd.DataFrame()
    
    print(f"Found {len(matched_data)} matched transaction pairs")
    
    SLIPPAGE = 0.005  # 0.5% slippage
    
    # BTC -> DOGE direction
    matched_data['btc_to_doge_cost'] = (
        matched_data['output_total_usd_btc'] * (1 + SLIPPAGE) +
        matched_data['fee_usd_btc'] +  # BTC fee
        matched_data['output_total_usd_doge'] * (1 + SLIPPAGE) +
        matched_data['fee_usd_doge']  # DOGE fee
    )
    
    matched_data['btc_to_doge_profit'] = (
        matched_data['output_total_usd_doge'] * (1 - SLIPPAGE) -
        matched_data['output_total_usd_btc'] * (1 + SLIPPAGE)
    )
    
    # DOGE -> BTC direction
    matched_data['doge_to_btc_cost'] = (
        matched_data['output_total_usd_doge'] * (1 + SLIPPAGE) +
        matched_data['fee_usd_doge'] +  # DOGE fee
        matched_data['output_total_usd_btc'] * (1 + SLIPPAGE) +
        matched_data['fee_usd_btc']  # BTC fee
    )
    
    matched_data['doge_to_btc_profit'] = (
        matched_data['output_total_usd_btc'] * (1 - SLIPPAGE) -
        matched_data['output_total_usd_doge'] * (1 + SLIPPAGE)
    )
    
    # Print profit analysis
    print("\nProfit Analysis:")
    print(f"Mean BTC->DOGE profit: ${matched_data['btc_to_doge_profit'].mean():.2f}")
    print(f"Mean DOGE->BTC profit: ${matched_data['doge_to_btc_profit'].mean():.2f}")
    
    # Select optimal direction
    matched_data['profit'] = np.maximum(
        matched_data['btc_to_doge_profit'],
        matched_data['doge_to_btc_profit']
    )
    
    matched_data['direction'] = np.where(
        matched_data['btc_to_doge_profit'] > matched_data['doge_to_btc_profit'],
        'BTC->DOGE',
        'DOGE->BTC'
    )
    
    # Calculate metadata
    matched_data['total_volume'] = matched_data['output_total_usd_btc'] + matched_data['output_total_usd_doge']
    matched_data['profit_percentage'] = (matched_data['profit'] / matched_data['total_volume']) * 100
    matched_data['hour'] = matched_data['time'].dt.hour
    
    # Filter profitable opportunities
    profitable_data = matched_data[matched_data['profit'] > 0].copy()
    
    if len(profitable_data) > 0:
        print(f"\nFound {len(profitable_data)} profitable opportunities")
        print("\nDirection breakdown:")
        print(profitable_data['direction'].value_counts())
        print(f"Average profit: ${profitable_data['profit'].mean():.2f}")
        print(f"Average profit percentage: {profitable_data['profit_percentage'].mean():.2f}%")
        print(f"Max profit: ${profitable_data['profit'].max():.2f}")
    else:
        print("\nNo profitable arbitrage opportunities found")
    
    return profitable_data


if __name__ == "__main__":
    try:
        print("Starting Blockchain Forensics Workflow...")

        # Load datasets
        bitcoin_data, bitcoin_date = load_dataset_by_keyword("bitcoin_transactions")
        dogecoin_data, dogecoin_date = load_dataset_by_keyword("dogecoin_transactions")

        if bitcoin_data is None or dogecoin_data is None:
            raise ValueError("Failed to load datasets.")

        # Preprocess data
        bitcoin_data = preprocess_data(bitcoin_data)
        dogecoin_data = preprocess_data(dogecoin_data)


        # Filter transactions
        print("Filtering transactions...")
        bitcoin_threshold = load_threshold("bitcoin")
        dogecoin_threshold = load_threshold("dogecoin")
        
        filtered_bitcoin_data = filter_transactions(bitcoin_data, bitcoin_threshold)
        filtered_dogecoin_data = filter_transactions(dogecoin_data, dogecoin_threshold)

        # Save filtered data
        for data, name, date in [
            (filtered_bitcoin_data, "bitcoin", bitcoin_date),
            (filtered_dogecoin_data, "dogecoin", dogecoin_date)
        ]:
            output_file = os.path.join(RESULTS_DIR, f"filtered_{name}_transactions_{date}.csv")
            data.to_csv(output_file, index=False)
            print(f"Filtered {name} transactions saved to {output_file}")

        # Block analysis
        print("\nPerforming block analysis...")
        block_stats_btc = cluster_by_block(filtered_bitcoin_data)
        block_stats_doge = cluster_by_block(filtered_dogecoin_data)
        
        high_risk_blocks_btc = detect_high_risk_blocks(block_stats_btc)
        high_risk_blocks_doge = detect_high_risk_blocks(block_stats_doge)

        # Extract high-risk transactions
        high_risk_btc_ids = high_risk_blocks_btc['block_id'].tolist()
        high_risk_doge_ids = high_risk_blocks_doge['block_id'].tolist()
        
        high_risk_bitcoin_data = filtered_bitcoin_data[filtered_bitcoin_data['block_id'].isin(high_risk_btc_ids)].copy()
        high_risk_dogecoin_data = filtered_dogecoin_data[filtered_dogecoin_data['block_id'].isin(high_risk_doge_ids)].copy()

        # Save high-risk data
        high_risk_bitcoin_file = os.path.join(RESULTS_DIR, f"high_risk_bitcoin_transactions_{bitcoin_date}.csv")
        high_risk_bitcoin_data.to_csv(high_risk_bitcoin_file, index=False)
        print(f"High-risk Bitcoin transactions saved to {high_risk_bitcoin_file}")

        # Visualizations
        print("\nGenerating visualizations...")
        
        # High-risk transactions visualization
        for data, name, date in [
            (high_risk_bitcoin_data, "bitcoin", bitcoin_date),
            (high_risk_dogecoin_data, "dogecoin", dogecoin_date)
        ]:
            figure_path = os.path.join(FIGURES_DIR, f'high_risk_{name}_transactions_{date}.png')
            visualize_high_risk_transactions(data, name, date, figure_path)

        # High-risk blocks visualization
        for blocks, name, date in [
            (high_risk_blocks_btc, "bitcoin", bitcoin_date),
            (high_risk_blocks_doge, "dogecoin", dogecoin_date)
        ]:
            figure_path = os.path.join(FIGURES_DIR, f'high_risk_blocks_{name}_{date}.png')
            visualize_high_risk_blocks(blocks, name, date, figure_path)

        # Cross-currency analysis
        print("\nPerforming cross-currency analysis...")
        if not high_risk_bitcoin_data.empty and not high_risk_dogecoin_data.empty:
            matched_data = match_cross_currency_transactions(high_risk_bitcoin_data, high_risk_dogecoin_data)
            exchange_rate = 254213.30271049644  
            
            # Anomaly detection
            print("Detecting anomalies...")
            anomalous_transactions = detect_anomalous_transactions(matched_data)
            if not anomalous_transactions.empty:
                anomalous_file = os.path.join(RESULTS_DIR, f"anomalous_transactions_{bitcoin_date}.csv")
                anomalous_transactions.to_csv(anomalous_file, index=False)
                
                anomalous_figure_path = os.path.join(FIGURES_DIR, f'anomalous_transactions_{bitcoin_date}.png')
                visualize_anomalies(matched_data, anomalous_transactions, 'Anomalous Transactions Detection', anomalous_figure_path)

            # Arbitrage analysis
            print("\nAnalyzing arbitrage opportunities...")
            arbitrage_opportunities = analyze_arbitrage_opportunities(
                high_risk_bitcoin_data, 
                high_risk_dogecoin_data
            )
            
            arbitrage_figure_path = os.path.join(FIGURES_DIR, f'arbitrage_opportunities_{bitcoin_date}_{dogecoin_date}.png')
            visualize_arbitrage_opportunities(arbitrage_opportunities, 'Cross-Currency Arbitrage Analysis', arbitrage_figure_path)

        print("\nAnalysis Complete. Results saved.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise