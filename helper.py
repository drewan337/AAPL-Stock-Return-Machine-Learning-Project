import pdb
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
import pickle
import json
from keras.models import load_model
from keras.utils import to_categorical

class Helper():
    def __init__(self, data_dir="./Data", model_dir="models", useKerasExt=True):
        self.DATA_DIR = data_dir
        self.model_dir = model_dir
        
        # Use '.keras' file extension for saving model (it saves everything)
        self.useKerasExt = useKerasExt
        
        self.sample_dir = os.path.join(self.DATA_DIR, "sample")
        self.train_dir = os.path.join(self.DATA_DIR, "train")
        
        # Create data directory if it doesn't exist
        if not os.path.isdir(self.DATA_DIR):
            print(f"Warning: Data directory '{self.DATA_DIR}' does not exist.")
        
        # Create model directory if it doesn't exist
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"Created model directory: {self.model_dir}")
    
    def get_available_tickers(self, data_dir=None):
        """
        Get list of available tickers in a data directory
        """
        if data_dir is None:
            data_dir = self.train_dir
        
        if not os.path.isdir(data_dir):
            print(f"Directory {data_dir} does not exist.")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        
        return sorted(tickers)
    
    def print_data_info(self):
        """
        Print information about available data directories and tickers
        """
        
        # Check train directory
        if os.path.isdir(self.train_dir):
            train_tickers = self.get_available_tickers(self.train_dir)
            print(f"\nTrain directory: {self.train_dir}")
            print(f"  Tickers available: {len(train_tickers)}")
            print(f"  Tickers: {', '.join(train_tickers)}")
            
            # Get date range from AAPL
            if 'AAPL' in train_tickers:
                df = pd.read_csv(os.path.join(self.train_dir, 'AAPL.csv'), 
                                index_col='Dt', parse_dates=True)
                print(f"  Date range (AAPL): {df.index.min().date()} to {df.index.max().date()}")
                print(f"  Number of rows (AAPL): {len(df)}")
        else:
            print(f"\nTrain directory not found: {self.train_dir}")
        
        # Check sample directory
        if os.path.isdir(self.sample_dir):
            sample_tickers = self.get_available_tickers(self.sample_dir)
            print(f"\nSample directory: {self.sample_dir}")
            print(f"  Tickers available: {len(sample_tickers)}")
            print(f"  Tickers: {', '.join(sample_tickers)}")
            
            # Get date range from AAPL
            if 'AAPL' in sample_tickers:
                df = pd.read_csv(os.path.join(self.sample_dir, 'AAPL.csv'), 
                                index_col='Dt', parse_dates=True)
                print(f"  Date range (AAPL): {df.index.min().date()} to {df.index.max().date()}")
                print(f"  Number of rows (AAPL): {len(df)}")
    
    def attrRename(self, df, ticker):
        """
        Rename attributes of DataFrame
        """
        rename_map = {orig: ticker + "_" + orig.replace(" ", "_") for orig in df.columns.to_list()}
        return df.rename(columns=rename_map)
    
    def getData(self, tickers, indx=None, attrs=None, data_dir=None, use_train=True):
        """
        Return DataFrame with data for a list of tickers plus an index
        """
        # Determine which directory to use
        if data_dir is None:
            data_dir = self.train_dir if use_train else self.sample_dir
        
        dateAttr = "Dt"
        
        # Convert single ticker to list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Read the CSV files
        dfs = []
        for ticker_num, ticker in enumerate(tickers):
            ticker_file = os.path.join(data_dir, "{t}.csv".format(t=ticker))
            
            if not os.path.exists(ticker_file):
                print(f"Warning: File {ticker_file} not found. Skipping {ticker}.")
                continue
            
            # Date as index
            if attrs:
                use_cols = attrs.copy()
                if dateAttr not in use_cols:
                    use_cols.insert(0, dateAttr)
                ticker_df = pd.read_csv(ticker_file, index_col=dateAttr, 
                                       parse_dates=True, usecols=use_cols)
            else:
                ticker_df = pd.read_csv(ticker_file, index_col=dateAttr, parse_dates=True)
            
            # Rename attributes with ticker name
            ticker_df = self.attrRename(ticker_df, ticker)
            dfs.append(ticker_df)
        
        # Add index data if provided
        if indx:
            index_file = os.path.join(data_dir, "{t}.csv".format(t=indx))
            if os.path.exists(index_file):
                if attrs:
                    use_cols = attrs.copy()
                    if dateAttr not in use_cols:
                        use_cols.insert(0, dateAttr)
                    index_df = pd.read_csv(index_file, index_col=dateAttr, 
                                          parse_dates=True, usecols=use_cols)
                else:
                    index_df = pd.read_csv(index_file, index_col=dateAttr, parse_dates=True)
                index_df = self.attrRename(index_df, indx)
                dfs.append(index_df)
            else:
                print(f"Warning: Index file {index_file} not found.")
        
        # Combine all dataframes
        if len(dfs) > 0:
            data_df = pd.concat(dfs, axis=1)
            return data_df
        else:
            raise ValueError("No data files were successfully loaded.")
    
    def getTrainData(self, tickers, indx=None, attrs=None):
        return self.getData(tickers, indx=indx, attrs=attrs, use_train=True)
    
    def getSampleData(self, tickers, indx=None, attrs=None):
        return self.getData(tickers, indx=indx, attrs=attrs, use_train=False)
    
    def renamePriceToRet(self, df, priceAttr="Adj Close"):
        """
        Rename columns from price attribute name to 'Ret'
        """
        rename_map = {orig: orig.replace(priceAttr.replace(" ", "_"), "Ret") 
                     for orig in df.columns.to_list()}
        return df.rename(columns=rename_map)
    
    def calculateReturns(self, df, price_col="Adj_Close", method="pct_change", prefix_pattern=None):
        """
        Calculate returns from price data
        
        Parameters:
        price_col: String
            Base name of price column (without ticker prefix)
        method: String
            Use 'pct_change' for percentage change or 'log' for log returns
        prefix_pattern: String (optional)
            Pattern to match ticker prefix (e.g., "AAPL")
            If None, calculates returns for all matching columns
        """
        df_returns = df.copy()
        
        # Find all columns that match the price pattern
        if prefix_pattern:
            price_columns = [col for col in df.columns 
                           if col.startswith(prefix_pattern) and price_col in col]
        else:
            price_columns = [col for col in df.columns if price_col in col]
        
        for col in price_columns:
            ticker = col.split("_")[0]
            ret_col = f"{ticker}_Ret"
            
            if method == "pct_change":
                df_returns[ret_col] = df[col].pct_change()
            elif method == "log":
                df_returns[ret_col] = np.log(df[col] / df[col].shift(1))
            else:
                raise ValueError("method must be 'pct_change' or 'log'")
        
        return df_returns
    
    def saveModel(self, model, model_path):
        """
        Save Keras model 
        """
        if self.useKerasExt:
            model_save_file = model_path + '.keras'
        else:
            model_save_file = model_path + '.h5'
            
        model.save(model_save_file)
        print(f"Model saved in {model_save_file}; submit with your assignment.")
        return model_save_file
    
    def loadModel(self, model_save_file):
        """
        Load Keras model
        """
        model = keras.models.load_model(model_save_file)
        print(f"Model loaded from {model_save_file}")
        return model
    
    def adjust_all_prices(self, df):
        """
        Returns DataFrame with these columns: Adj_Factor, Adj_High, Adj_Low, Adj_Open
        Removes non adjusted columns and Adj_Factor
        """
        df = df.copy()
        
        # Get list of columns
        columns = df.columns.tolist()
    
        if 'Adj_Close' in columns:
            close_col = 'Close'
            adj_close_col = 'Adj_Close'
            high_col = 'High'
            low_col = 'Low'
            open_col = 'Open'
            adj_factor_col = 'Adj_Factor'
            adj_high_col = 'Adj_High'
            adj_low_col = 'Adj_Low'
            adj_open_col = 'Adj_Open'
        
        # Calculate adjustment factor
        df[adj_factor_col] = df[adj_close_col] / df[close_col] # type: ignore
        
        # Apply to all prices
        df[adj_high_col] = df[high_col] * df[adj_factor_col] # type: ignore
        df[adj_low_col] = df[low_col] * df[adj_factor_col] # type: ignore
        df[adj_open_col] = df[open_col] * df[adj_factor_col] # type: ignore
        
        return df
    
    def calculate_acf(self, series, nlags=50):
        """
        Calculate autocorrelation function
        """
        series = series.dropna()
        mean = series.mean()
        c0 = np.sum((series - mean) ** 2) / len(series)
        
        acf_values = [1.0]  # Lag 0 is always 1
        for k in range(1, nlags + 1):
            c_k = np.sum((series[:-k] - mean) * (series[k:] - mean)) / len(series)
            acf_values.append(c_k / c0)
        
        return np.array(acf_values)
    
    def plot_acf(self, series, nlags=50, title='Autocorrelation Function', figsize=(15, 6)):
        """
        Plot autocorrelation function with confidence intervals
        """
        acf_values = self.calculate_acf(series, nlags)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars
        lags = np.arange(len(acf_values))
        ax.bar(lags, acf_values, width=0.8, alpha=0.7, edgecolor='black')
        
        # Add confidence interval (95%)
        confidence_interval = 1.96 / np.sqrt(len(series.dropna()))
        ax.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('ACF', fontsize=12)
        ax.set_xlim(-1, nlags + 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return acf_values, confidence_interval
    
    def analyze_volatility_clustering(self, df, return_col='Return', figsize=(18, 14)):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import linregress, pearsonr
        
        # Calculate returns and related metrics
        returns = df[return_col].dropna()
        returns_squared = returns ** 2
        abs_returns = np.abs(returns)
        
        # Calculate rolling volatility
        vol_20d = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized (252 trading days in 1 year)
        vol_60d = returns.rolling(window=60).std() * np.sqrt(252) * 100
        
        fig = plt.figure(figsize=figsize)
        
        # Plot 1: Returns Over Time
        ax1 = plt.subplot(4, 1, 1)
        _=ax1.plot(returns.index, returns * 100, linewidth=0.5, alpha=0.7, color='steelblue')
        _=ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        _=ax1.fill_between(returns.index, 0, returns * 100, 
                        where=(returns > 0), alpha=0.3, color='green', label='Positive')
        _=ax1.fill_between(returns.index, 0, returns * 100, 
                        where=(returns < 0), alpha=0.3, color='red', label='Negative')
        
        # Highlight high volatility periods
        high_vol_periods = [
            ('2000-09-01', '2003-03-01', 'Dot-com Crash'),
            ('2008-01-01', '2009-06-01', 'Financial Crisis')
        ]
        for start, end, label in high_vol_periods:
            ax1.axvspan(start, end, alpha=0.15, color='yellow', zorder=0) # type: ignore
            
        _=ax1.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
        _=ax1.set_title('Daily Returns (Volatility clusters highlighted)', 
                    fontsize=14, fontweight='bold')
        _=ax1.legend(loc='upper right', fontsize=10)
        _=ax1.grid(True, alpha=0.3)
        
        # Plot 2: Absolute Returns & Rolling Volatility
        ax2 = plt.subplot(4, 1, 2)
        _=ax2.plot(abs_returns.index, abs_returns * 100, 
                linewidth=0.8, alpha=0.7, color='purple', label='Absolute Returns')
        _=ax2.plot(vol_20d.index, vol_20d, 
                linewidth=2, alpha=0.8, color='red', label='20-day Rolling Vol')
        
        for start, end, label in high_vol_periods:
            _=ax2.axvspan(start, end, alpha=0.15, color='yellow', zorder=0) # type: ignore
            mid_date = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            _=ax2.text(mid_date, ax2.get_ylim()[1] * 0.9, label, # type: ignore
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        _=ax2.set_ylabel('Volatility (%)', fontsize=11, fontweight='bold')
        _=ax2.set_title('Absolute Returns & Rolling Volatility - High Vol Periods Cluster Together', 
                    fontsize=14, fontweight='bold')
        _=ax2.legend(loc='upper right', fontsize=10)
        _=ax2.grid(True, alpha=0.3)
        
        # Plot 3: ACF of Returns
        _=ax3 = plt.subplot(4, 2, 5)
        acf_returns = self.calculate_acf(returns, nlags=80)
        lags = np.arange(len(acf_returns))
        conf_int = 1.96 / np.sqrt(len(returns))
        
        _=ax3.bar(lags, acf_returns, width=0.8, alpha=0.7, edgecolor='black', color='blue')
        _=ax3.axhline(y=conf_int, color='red', linestyle='--', linewidth=1, alpha=0.5)
        _=ax3.axhline(y=-conf_int, color='red', linestyle='--', linewidth=1, alpha=0.5)
        _=ax3.axhline(y=0, color='black', linewidth=0.5)
        _=ax3.set_xlabel('Lag', fontsize=11)
        _=ax3.set_ylabel('ACF', fontsize=11)
        _=ax3.set_title('ACF of Returns', fontsize=12, fontweight='bold')
        _=ax3.grid(True, alpha=0.3, axis='y')
        _=ax3.set_xlim(-1, 80)
        
        # Plot 4: ACF of Squared Returns
        _=ax4 = plt.subplot(4, 2, 6)
        acf_squared = self.calculate_acf(returns_squared, nlags=80)
        
        _=ax4.bar(lags, acf_squared, width=0.8, alpha=0.7, edgecolor='black', color='darkred')
        _=ax4.axhline(y=conf_int, color='red', linestyle='--', linewidth=1, alpha=0.5)
        _=ax4.axhline(y=-conf_int, color='red', linestyle='--', linewidth=1, alpha=0.5)
        _=ax4.axhline(y=0, color='black', linewidth=0.5)
        _=ax4.set_xlabel('Lag', fontsize=11)
        _=ax4.set_ylabel('ACF', fontsize=11)
        _=ax4.set_title('ACF of Squared Returns (Volatility Clustering)', 
                    fontsize=12, fontweight='bold')
        _=ax4.grid(True, alpha=0.3, axis='y')
        _=ax4.set_xlim(-1, 80)
        
        # Plot 5: Scatter - Today's vs Yesterday's Volatility
        ax5 = plt.subplot(4, 2, 7)
        abs_returns_lag = abs_returns.shift(1)
        valid_mask = ~(abs_returns.isna() | abs_returns_lag.isna())
        
        _=ax5.scatter(abs_returns_lag[valid_mask] * 100, abs_returns[valid_mask] * 100, 
                alpha=0.3, s=10, color='purple')
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = linregress(
            abs_returns_lag[valid_mask] * 100, abs_returns[valid_mask] * 100
        )
        x_line = np.linspace(0, abs_returns_lag.max() * 100, 100)
        y_line = slope * x_line + intercept
        ax5.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8, label=f'R² = {r_value**2:.3f}') # type: ignore
        
        _=ax5.set_xlabel('|Return| Yesterday (%)', fontsize=11)
        _=ax5.set_ylabel('|Return| Today (%)', fontsize=11)
        _=ax5.set_title('Volatility Persistence - Yesterday\'s Vol Predicts Today\'s', 
                    fontsize=12, fontweight='bold')
        _=ax5.legend(fontsize=10)
        _=ax5.grid(True, alpha=0.3)
        
        # Plot 6: Volatility Regimes
        ax6 = plt.subplot(4, 2, 8)
        vol_20d_clean = vol_20d.dropna()
        low_vol = vol_20d_clean < vol_20d_clean.quantile(0.33)
        med_vol = (vol_20d_clean >= vol_20d_clean.quantile(0.33)) & (vol_20d_clean < vol_20d_clean.quantile(0.67))
        high_vol = vol_20d_clean >= vol_20d_clean.quantile(0.67)
        
        _=ax6.scatter(vol_20d_clean[low_vol].index, vol_20d_clean[low_vol], 
                c='green', alpha=0.5, s=5, label='Low Vol')
        _=ax6.scatter(vol_20d_clean[med_vol].index, vol_20d_clean[med_vol], 
                c='orange', alpha=0.5, s=5, label='Medium Vol')
        _=ax6.scatter(vol_20d_clean[high_vol].index, vol_20d_clean[high_vol], 
                c='red', alpha=0.5, s=5, label='High Vol')
        
        _=ax6.set_ylabel('Annualized Volatility (%)', fontsize=11)
        _=ax6.set_xlabel('Date', fontsize=11)
        _=ax6.set_title('Volatility Regimes - Stays in Same Regime for Extended Periods', 
                    fontsize=12, fontweight='bold')
        _=ax6.legend(fontsize=10)
        _=ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Volatility persistence
        vol_persistence = abs_returns.corr(abs_returns.shift(1))
        
        # Volatility regimes
        regime_stats = {
            'low_vol_mean': vol_20d_clean[low_vol].mean(),
            'med_vol_mean': vol_20d_clean[med_vol].mean(),
            'high_vol_mean': vol_20d_clean[high_vol].mean(),
            'low_vol_days': low_vol.sum(),
            'med_vol_days': med_vol.sum(),
            'high_vol_days': high_vol.sum()
        }
        
        # ARCH test
        returns_clean = returns.dropna()
        returns_sq = returns_clean ** 2
        returns_sq_lag = returns_sq.shift(1).dropna()
        returns_sq_align = returns_sq[returns_sq_lag.index]
        arch_corr, arch_pvalue = pearsonr(returns_sq_lag, returns_sq_align)
        
        print("\n Autocorrelation:")
        print(f"   ACF of returns at lag 1: {acf_returns[1]:.4f}")
        print(f"   ACF of squared returns at lag 1: {acf_squared[1]:.4f}")
        
        print("\n Volatility Persistence:")
        print(f"   Correlation (|return_t| vs |return_t-1|): {vol_persistence:.4f}")
        print(f"   R² from scatter plot: {r_value**2:.4f}") # type: ignore
        
        print("\n Volatility Regimes:")
        print(f"   Low vol regime (<33%): Mean = {regime_stats['low_vol_mean']:.1f}%")
        print(f"   Medium vol regime: Mean = {regime_stats['med_vol_mean']:.1f}%")
        print(f"   High vol regime (>67%): Mean = {regime_stats['high_vol_mean']:.1f}%")
        print(f"   Days in each regime: {regime_stats['low_vol_days']}, {regime_stats['med_vol_days']}, {regime_stats['high_vol_days']}")
        
        print("\n ARCH effects (ARCH LM Test):")
        print(f"   Correlation (return²_t vs return²_t-1): {arch_corr:.4f}")
        print(f"   P-value: {arch_pvalue:.6f}")
        print(f"   Strong if arch_pvalue < 0.001 else Weak") # type: ignore
        
        # Return statistics
        return {
            'acf_returns': acf_returns,
            'acf_squared': acf_squared,
            'vol_persistence': vol_persistence,
            'regime_stats': regime_stats,
            'arch_corr': arch_corr,
            'arch_pvalue': arch_pvalue,
            'r_squared': r_value**2 # type: ignore
        }
    
    def create_features(self, df, lookback_returns=5, lookback_vol=3, is_sequence=False):
        """
        Create features for both FCNN and sequence models
        
        Parameters:
        is_sequence : bool
            False (default): Create lagged features for FCNN/CNN (17 lagged features)
            True: Create unlagged features for sequences (17 unlagged features)
            
        (NaN rows NOT dropped)
        """
        import numpy as np
        import pandas as pd
        
        df_features = df.copy()

        # 1. Basic returns
        if 'Return' not in df_features.columns:
            df_features['Return'] = df_features['Adj_Close'].pct_change()
        
        # 2. Lagged returns (FCNN only)
        if not is_sequence:
            for i in range(1, lookback_returns + 1):
                df_features[f'Return_lag_{i}'] = df_features['Return'].shift(i)

        # 3. Absolute returns
        df_features['Abs_Return'] = np.abs(df_features['Return'])
        
        if not is_sequence:
            for i in range(1, lookback_vol + 1):
                df_features[f'Abs_Return_lag_{i}'] = df_features['Abs_Return'].shift(i)
            
        # 4. Squared returns
        df_features['Return_squared'] = df_features['Return'] ** 2
        
        if not is_sequence:
            df_features['Return_squared_lag_1'] = df_features['Return_squared'].shift(1)
        
        # 5. Volatility features
        df_features['Vol_20d'] = df_features['Return'].rolling(window=20).std()
        df_features['Vol_60d'] = df_features['Return'].rolling(window=60).std()
        
        if not is_sequence:
            df_features['Vol_20d_lag_1'] = df_features['Vol_20d'].shift(1)
        
        # Volatility ratio
        df_features['Vol_Ratio'] = df_features['Vol_20d'] / (df_features['Vol_60d'] + 1e-10)
        
        if not is_sequence:
            df_features['Vol_Ratio_lag_1'] = df_features['Vol_Ratio'].shift(1)

        # 6. Volatility change
        df_features['Vol_Change'] = df_features['Vol_20d'] - df_features['Vol_20d'].shift(5)
        
        if not is_sequence:
            df_features['Vol_Change_lag_1'] = df_features['Vol_Change'].shift(1)

        # 7. Volume features
        df_features['Volume_Std'] = (
            (df_features['Volume'] - df_features['Volume'].mean()) / 
            (df_features['Volume'].std() + 1e-10)
        )
        
        if not is_sequence:
            df_features['Volume_Std_lag_1'] = df_features['Volume_Std'].shift(1)

        # 8. Volume × Volatility interaction (FCNN/CNN  only - requires lagged features)
        if not is_sequence:
            df_features['Vol_Volume_Interaction'] = (
                df_features['Vol_20d_lag_1'] * df_features['Volume_Std_lag_1']
            )

        # 9. Return × Volatility interaction (FCNN/CNN  only - requires lagged features)
        if not is_sequence:
            df_features['Return_Vol_Interaction'] = (
                df_features['Return_lag_1'] * df_features['Vol_20d_lag_1']
            )

        # 10. Momentum feature (5-day cumulative return)
        if not is_sequence:
            df_features['Return_5d_sum'] = df_features['Return'].shift(1).rolling(window=5).sum()
        else:
            # For sequences, calculate without initial lag (sequence will handle time dimension)
            df_features['Return_5d_sum'] = df_features['Return'].rolling(window=5).sum()

        # 11. Trend indicator
        df_features['MA_50'] = df_features['Adj_Close'].rolling(window=50).mean()
        df_features['Price_MA50_Ratio'] = df_features['Adj_Close'] / (df_features['MA_50'] + 1e-10)
        
        if not is_sequence:
            df_features['Price_MA50_Ratio_lag_1'] = df_features['Price_MA50_Ratio'].shift(1)


        # Drop features based on mode
        if not is_sequence:
            # Drop original features
            original_features = ['Adj_Close', 'Volume', 'Close', 'Div', 'Factor', 'High', 'Low', 'Open']
            df_features = df_features.drop(columns=original_features, axis='columns')
            # FCNN/CNN : Drop unlagged features, keep only lagged versions
            temporary_features = ['Abs_Return', 'Return_squared', 'Vol_20d', 'Vol_60d', 
                                'Vol_Ratio', 'Vol_Change', 'Volume_Std', 'MA_50', 'Price_MA50_Ratio']
            df_features = df_features.drop(columns=temporary_features, axis='columns')
        else:
            # Drop original features but not the adjusted close and volume
            original_features = ['Close', 'Div', 'Factor', 'High', 'Low', 'Open']
            df_features = df_features.drop(columns=original_features, axis='columns')
            # SEQUENCES: Keep unlagged features, drop MA_50 (already have Price_MA50_Ratio)
            df_features = df_features.drop(columns=['MA_50'], axis='columns')
        
        return df_features
        
    def analyze_features(self, df, target_col='Return', top_n=10):
        """  
        Analyzes:
        1. Correlation with target and p-value test
        2. Top N most predictive features
        3. Feature categories comparison
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import stats
        
        # Define features to analyze (exclude target and features)
        exclude_cols = [target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate correlations and p-values
        correlations = {}
        p_values = {}
        
        for feat in feature_cols:
            # Drop NaN for this pair
            valid_data = df[[feat, target_col]].dropna()
            
            if len(valid_data) > 0:
                corr, p_val = stats.pearsonr(valid_data[feat], valid_data[target_col])
                correlations[feat] = corr
                p_values[feat] = p_val
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Feature': list(correlations.keys()),
            'Correlation': list(correlations.values()),
            'Abs_Correlation': [abs(c) for c in correlations.values()],
            'P_Value': list(p_values.values())
        }).sort_values('Abs_Correlation', ascending=False)
        
        # Determine significance
        results_df['Significant'] = results_df['P_Value'] < 0.05
        
        # Print top features
        print(f"\n{'Rank':<6} {'Feature':<30} {'Correlation':<12} {'P-Value':<12}")
        
        for i, row in results_df.head(top_n).iterrows():
            print(f"{results_df.index.get_loc(i)+1:<6} {row['Feature']:<30} {row['Correlation']:>7.4f} {row['P_Value']:>8.4f}") # type: ignore
        
        
        categories = {
            'Lagged Returns': [f for f in feature_cols if 'Return_lag' in f],
            'Lagged Abs Returns': [f for f in feature_cols if 'Abs_Return_lag' in f],
            'Volatility': [f for f in feature_cols if 'Vol' in f],
            'Volume': [f for f in feature_cols if 'Volume' in f],
            'Momentum': [f for f in feature_cols if 'sum' in f or 'ROC' in f],
            'Trend': [f for f in feature_cols if 'MA' in f or 'Price' in f]
        }
        
        # Visualization 
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Top features bar chart
        top_features = results_df.head(top_n)
        colors = ['green' if c > 0 else 'red' for c in top_features['Correlation']]
        
        axes[0].barh(range(len(top_features)), top_features['Correlation'], color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['Feature'], fontsize=10)
        axes[0].axvline(x=0, color='black', linewidth=1.5)
        axes[0].set_xlabel('Correlation with Return', fontsize=12)
        axes[0].set_title(f'Top {top_n} Most Predictive Features', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        # Plot 2: Category comparison
        category_data = []
        category_names = []
        for category, feats in categories.items():
            if feats:
                avg_corr = results_df[results_df['Feature'].isin(feats)]['Abs_Correlation'].mean()
                category_data.append(avg_corr)
                category_names.append(category)
        
        axes[1].barh(category_names, category_data, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('Average |Correlation|', fontsize=12)
        axes[1].set_title('Feature Category Importance', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()

    def prepare_features(self, ticker, lookback_returns=5, lookback_vol=3, use_train=True, is_sequence = False):
        """
        Load data, adjust prices, and create features
        
        This function prepares features for a single ticker. You can call it multiple
        times for different tickers, then join the results before training.
        
        Parameters:
        ticker : str
            Stock ticker to process (ex: 'AAPL', 'SPY')
        lookback_returns : int
            Number of lagged returns to create (default: 5)
        lookback_vol : int
            Number of lagged absolute returns to create (default: 3)
        is_sequence: bool
            indicates whether to use lagged features for non sequence models or non lagged features for sequence model
        """
        import pandas as pd
        
        # Load Data
        if use_train:
            data_raw = self.getTrainData(ticker)
        else:
            data_raw = self.getSampleData(ticker)

        # Adjust prices (using existing adjust_all_prices)
        # Remove ticker prefix in feature name
        data_raw.columns = [col.replace((ticker + '_'), '') for col in data_raw.columns]
        
        data_adj = self.adjust_all_prices(data_raw)

        # Create features (using existing create_features)
        features_df = self.create_features(data_adj, 
                                        lookback_returns=lookback_returns, 
                                        lookback_vol=lookback_vol, is_sequence=is_sequence)

        # Drop NaN rows and adjusted features for non sequence
        if is_sequence == False:
            adj_feature = ['Adj_Factor', 'Adj_High', 'Adj_Low', 'Adj_Open']
            features_df = features_df.dropna()
            features_df = features_df.drop(columns=adj_feature, axis='columns')
            
        # Drop NaN ros and add_factor for sequence
        else:
            adj_feature = ['Adj_Factor']
            features_df = features_df.dropna()
            features_df = features_df.drop(columns=adj_feature, axis='columns')
        
        return features_df


    def prepare_for_training(self, 
                            train_features_df,      # From train folder (2000-2016)
                            sample_features_df,     # From sample folder (2017)
                            target_col='Return',
                            val_ratio=0.15,         # Validation from train folder
                            save_artifacts=True,
                            ticker_name='AAPL'):
        """
        Prepare features for neural network training
        
        Parameters:
        train_features_df : DataFrame
            Features from train folder (2000-2016)
        sample_features_df : DataFrame
            Features from sample folder (2017)
        target_col : str
            Name of target column (default: 'Return')
        val_ratio : float
            Proportion of train data for validation (default: 0.15)
            Options: 0.10 (90/10 split) or 0.15 (85/15 split)
        save_artifacts : bool
            Whether to save scaler and metadata
        ticker_name : str
            Name for saving artifacts
        """
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import pickle
        
        # Check target column exists
        if target_col not in train_features_df.columns:
            raise ValueError(f"Target '{target_col}' not in train_features_df")
        if target_col not in sample_features_df.columns:
            raise ValueError(f"Target '{target_col}' not in sample_features_df")
        
        # Get feature columns
        feature_cols = [col for col in train_features_df.columns if col != target_col]
        
        # Split train folder into train/val
        train_size = int((1 - val_ratio) * len(train_features_df))
        train_data = train_features_df.iloc[:train_size]
        val_data = train_features_df.iloc[train_size:]
        
        # Use entire sample folder as test (test data has 251 days which is close to 200 days of prediction in holdout data)
        test_data = sample_features_df
        
        # Store date ranges
        train_dates = (train_data.index[0], train_data.index[-1])
        val_dates = (val_data.index[0], val_data.index[-1])
        test_dates = (test_data.index[0], test_data.index[-1])
        
        # Extract X and y
        X_train = train_data[feature_cols].values
        X_val = val_data[feature_cols].values
        X_test = test_data[feature_cols].values
        
        y_train = train_data[target_col].values
        y_val = val_data[target_col].values
        y_test = test_data[target_col].values
        
        # Check for invalid values
        has_nan = np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any()
        has_inf = np.isinf(X_train).any() or np.isinf(X_val).any() or np.isinf(X_test).any()
        
        if has_nan or has_inf:
            raise ValueError("NaN or Inf values detected!")
        
        # Standardize features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save artifacts
        if save_artifacts:
            with open(f'scaler_{ticker_name}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open(f'feature_names_{ticker_name}.pkl', 'wb') as f:
                pickle.dump(feature_cols, f)
        
        return {
            'X_train_scaled': X_train_scaled,
            'y_train': y_train,
            'X_val_scaled': X_val_scaled,
            'y_val': y_val,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': feature_cols,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates
        }
        
    def prepare_sequences_for_training(self, X_train_seq, y_train_seq,
                                   X_sample_seq, y_sample_seq,
                                   val_ratio=0.15):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Split
        train_size = int((1 - val_ratio) * len(X_train_seq))
        X_train = X_train_seq[:train_size].copy()
        y_train = y_train_seq[:train_size]
        X_val = X_train_seq[train_size:].copy()
        y_val = y_train_seq[train_size:]
        X_test = X_sample_seq.copy()
        y_test = y_sample_seq
        
        # Scale each feature
        n_samples_train, lookback, n_features = X_train.shape
        n_samples_val = X_val.shape[0]  # Might be 0!
        n_samples_test = X_test.shape[0]
        
        feature_scalers = []
        
        for feat_idx in range(n_features):
            scaler = StandardScaler()
            
            # Train
            train_feat = X_train[:, :, feat_idx].reshape(-1, 1)
            scaler.fit(train_feat)
            X_train[:, :, feat_idx] = scaler.transform(train_feat).reshape(n_samples_train, lookback)
            
            # Val (skip if empty!) 
            if n_samples_val > 0:
                val_feat = X_val[:, :, feat_idx].reshape(-1, 1)
                X_val[:, :, feat_idx] = scaler.transform(val_feat).reshape(n_samples_val, lookback)
            
            # Test
            test_feat = X_test[:, :, feat_idx].reshape(-1, 1)
            X_test[:, :, feat_idx] = scaler.transform(test_feat).reshape(n_samples_test, lookback)
            
            feature_scalers.append(scaler)
        
        print(f"mean≈{X_train.mean():.2e}, std≈{X_train.std():.2f}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_scalers': feature_scalers
        }
        
    def calc_metrics(self, y_true, y_pred):
        from sklearn.metrics import mean_absolute_error
        from scipy.stats import spearmanr
        import numpy as np
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # Huber Loss
        delta=0.01
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = np.mean(0.5 * quadratic**2 + delta * linear)
        
        
        # Directional Accuracy
        dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean()
        
        # Information Coefficient (with safety check)
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            # Constant array - correlation undefined
            ic = 0.0
            ic_p = 1.0
            print("Warning: Constant predictions or actuals - IC set to 0")
        else:
            try:
                ic, ic_p = spearmanr(y_true, y_pred)
                # Handle NaN results
                if np.isnan(ic): # type: ignore
                    ic = 0.0
                    ic_p = 1.0
            except:
                ic = 0.0
                ic_p = 1.0
        
        # Large move metrics
        threshold = 2 * np.std(y_true)
        
        # If no variance in y_true, can't define large moves
        if threshold < 1e-8:
            precision = 0.0
            recall = 0.0
        else:
            true_large = np.abs(y_true) > threshold 
            pred_large = np.abs(y_pred) > threshold
            
            if pred_large.sum() > 0:
                precision = (true_large & pred_large).sum() / pred_large.sum()
            else:
                precision = 0.0
            
            if true_large.sum() > 0:
                recall = (true_large & pred_large).sum() / true_large.sum()
            else:
                recall = 0.0
        
        return {
            'mae': mae,
            'huber_loss': huber,
            'directional_accuracy': dir_acc,
            'ic': ic,
            'ic_pvalue': ic_p,
            'precision_large_moves': precision,
            'recall_large_moves': recall
        }
        
    def compare_models(self, models_dict, data, plot=True):
        """
        Train and compare multiple non sequential neural network models
        Does error analysis on a dictionary containing various models
        Metrics:
        - MAE: Mean Absolute Error
        - Huber Loss: Robust to outliers
        - Directional Accuracy: % of correct sign predictions
        - Information Coefficient (IC): Spearman correlation
        - Precision/Recall for large moves: Performance on significant returns
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from keras.callbacks import EarlyStopping
        
        results = {}
        
        print(f"\nModels to compare: {len(models_dict)}")
        print(f"Training samples: {len(data['X_train_scaled'])}")
        print(f"Validation samples: {len(data['X_val_scaled'])}")
        print(f"Test samples: {len(data['X_test_scaled'])}")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train each model
        for model_name, model in models_dict.items():
            print(f"Training: {model_name}")
            
            # Count parameters
            total_params = model.count_params()
            print(f"\nTotal parameters: {total_params:,}")
            
            # Train
            history = model.fit(
                data['X_train_scaled'], data['y_train'],
                validation_data=(data['X_val_scaled'], data['y_val']),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            epochs_trained = len(history.history['loss'])
            
            # Predictions
            y_train_pred = model.predict(data['X_train_scaled'], verbose=0).flatten()
            y_val_pred = model.predict(data['X_val_scaled'], verbose=0).flatten()
            y_test_pred = model.predict(data['X_test_scaled'], verbose=0).flatten()
            
            # Calculate comprehensive metrics
            
            train_metrics = self.calc_metrics(data['y_train'], y_train_pred) 
            val_metrics = self.calc_metrics(data['y_val'], y_val_pred)
            test_metrics = self.calc_metrics(data['y_test'], y_test_pred) 
            
            # Store results
            results[model_name] = {
                'model': model,
                'history': history,
                'epochs': epochs_trained,
                'params': total_params,
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred,
                    'test': y_test_pred
                }
            }
            
            # Print test metrics
            print(f"\nTest Performance:")
            print(f"  MAE:              {test_metrics['mae']*100:.4f}%")
            print(f"  Huber Loss:       {test_metrics['huber_loss']*100:.4f}%")
            print(f"  Directional Acc:  {test_metrics['directional_accuracy']*100:.2f}%")
            print(f"  IC (Spearman):    {test_metrics['ic']:.4f} (p={test_metrics['ic_pvalue']:.4f})")
            print(f"  Precision (>1%):  {test_metrics['precision_large_moves']*100:.2f}%")
            print(f"  Recall (>1%):     {test_metrics['recall_large_moves']*100:.2f}%")
        
        # Create comparison table
        print("Test set comparison sumamry")
        
        comparison_data = []
        for name, res in results.items():
            comparison_data.append({
                'Model': name,
                'Params': res['params'],
                'Epochs': res['epochs'],
                'MAE%': res['test']['mae'] * 100,
                'Huber%': res['test']['huber_loss'] * 100,
                'Dir_Acc%': res['test']['directional_accuracy'] * 100,
                'IC': res['test']['ic'],
                'Prec%': res['test']['precision_large_moves'] * 100,
                'Recall%': res['test']['recall_large_moves'] * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by directional accuracy (most important for trading)
        comparison_df = comparison_df.sort_values('Dir_Acc%', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        
        # Overfitting check
        print("Overfitting check")
        for name, res in results.items():
            train_mae = res['train']['mae']
            test_mae = res['test']['mae']
            
            # Safe absolute difference (in percentage points)
            gap_abs = (test_mae - train_mae) * 100
            
            # Safe relative change
            if train_mae > 1e-6:
                gap_rel = ((test_mae - train_mae) / train_mae) * 100
                gap_rel = np.clip(gap_rel, -100, 200)  # Cap extremes
            else:
                gap_rel = 0.0
            
            # Determine status using absolute difference (more stable)
            if gap_abs > 0.5:  # 0.5 percentage points
                status = "Overfitting"
            elif gap_abs > 0.2:  # 0.2 percentage points
                status = "Slight overfitting"
            elif gap_abs < -0.2:
                status = "Test better than train (investigate)"
            else:
                status = "Good generalization"
    
            print(f"  {name}:")
            print(f"    Train MAE = {train_mae*100:.4f}%, Test MAE = {test_mae*100:.4f}%")
            print(f"    Gap = {gap_abs:+.4f} pct pts ({gap_rel:+.1f}%) {status}")
        
        # Visualization
        if plot:
            n_models = len(results)
            fig = plt.figure(figsize=(16, 12))
            
            model_names = list(results.keys())
            
            # Plot 1: Key Metrics Comparison
            ax1 = plt.subplot(3, 2, 1)
            x = np.arange(n_models)
            width = 0.25
            
            mae_vals = [results[m]['test']['mae'] * 100 for m in model_names]
            dir_vals = [results[m]['test']['directional_accuracy'] * 100 for m in model_names]
            ic_vals = [results[m]['test']['ic'] * 100 for m in model_names]  # Scale IC by 100 for visibility
            
            ax1.bar(x - width, mae_vals, width, label='MAE (%)', alpha=0.8, color='coral')
            ax1.bar(x, dir_vals, width, label='Dir Acc (%)', alpha=0.8, color='skyblue')
            ax1.bar(x + width, ic_vals, width, label='IC (×100)', alpha=0.8, color='lightgreen')
            
            ax1.set_xlabel('Model', fontsize=11)
            ax1.set_ylabel('Value', fontsize=11)
            ax1.set_title('Key Metrics Comparison (Test Set)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Directional Accuracy
            ax2 = plt.subplot(3, 2, 2)
            dir_acc_vals = [results[m]['test']['directional_accuracy'] * 100 for m in model_names]
            colors = ['green' if v > 52 else 'orange' if v > 50 else 'red' for v in dir_acc_vals]
            
            ax2.barh(model_names, dir_acc_vals, alpha=0.8, color=colors)
            ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
            ax2.axvline(x=52, color='orange', linestyle='--', linewidth=2, label='Acceptable (52%)')
            ax2.axvline(x=55, color='green', linestyle='--', linewidth=2, label='Excellent (55%)')
            
            ax2.set_xlabel('Directional Accuracy (%)', fontsize=11)
            ax2.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Precision vs Recall (Large Moves)
            ax3 = plt.subplot(3, 2, 3)
            prec_vals = [results[m]['test']['precision_large_moves'] * 100 for m in model_names]
            rec_vals = [results[m]['test']['recall_large_moves'] * 100 for m in model_names]
            
            ax3.scatter(rec_vals, prec_vals, s=200, alpha=0.6)
            for i, name in enumerate(model_names):
                ax3.annotate(name, (rec_vals[i], prec_vals[i]), fontsize=8, ha='right')
            
            ax3.set_xlabel('Recall (%) - Catches large moves', fontsize=11)
            ax3.set_ylabel('Precision (%) - Avoids false alarms', fontsize=11)
            ax3.set_title('Precision vs Recall for Large Moves (>1%)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Information Coefficient
            ax4 = plt.subplot(3, 2, 4)
            ic_vals = [results[m]['test']['ic'] for m in model_names]
            colors = ['green' if v > 0.03 else 'orange' if v > 0.01 else 'red' for v in ic_vals]
            
            ax4.barh(model_names, ic_vals, alpha=0.8, color=colors)
            ax4.axvline(x=0, color='black', linewidth=1)
            ax4.axvline(x=0.03, color='green', linestyle='--', linewidth=1, label='Good (0.03)')
            ax4.axvline(x=0.01, color='orange', linestyle='--', linewidth=1, label='Weak (0.01)')
            
            ax4.set_xlabel('Information Coefficient (Spearman)', fontsize=11)
            ax4.set_title('Information Coefficient - Rank Correlation', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Plot 5: Training History (Validation Loss)
            ax5 = plt.subplot(3, 2, 5)
            for name in model_names:
                ax5.plot(results[name]['history'].history['val_loss'], 
                        label=name, alpha=0.7, linewidth=1.5)
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('Validation Loss (MSE)', fontsize=11)
            ax5.set_title('Validation Loss During Training', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')  # Log scale for better visualization
            
            # Plot 6: MAE Comparison (Train vs Val vs Test)
            ax6 = plt.subplot(3, 2, 6)
            x_pos = np.arange(n_models)
            width = 0.25
            
            train_mae = [results[m]['train']['mae'] * 100 for m in model_names]
            val_mae = [results[m]['val']['mae'] * 100 for m in model_names]
            test_mae = [results[m]['test']['mae'] * 100 for m in model_names]
            
            ax6.bar(x_pos - width, train_mae, width, label='Train', alpha=0.8, color='blue')
            ax6.bar(x_pos, val_mae, width, label='Val', alpha=0.8, color='orange')
            ax6.bar(x_pos + width, test_mae, width, label='Test', alpha=0.8, color='red')
            
            ax6.set_xlabel('Model', fontsize=11)
            ax6.set_ylabel('MAE (%)', fontsize=11)
            ax6.set_title('MAE: Train vs Val vs Test', fontsize=12, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
        
        # Add comparison DataFrame to results
        results['comparison_df'] = comparison_df

        return results
    
    def compare_sequence_models(self, models_dict, data, plot=True):
        """
        Train and compare multiple sequence models (LSTM/GRU/TCN/Transformer)
        Similar to compare_models() but compatable with sequence data with shape (samples, timesteps, features)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from keras.callbacks import EarlyStopping
        
        results = {}
        
        print(f"\nSequence Models to compare: {len(models_dict)}")
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        print(f"Test samples: {len(data['X_test'])}")
        print(f"Sequence shape: {data['X_train'].shape} (samples, timesteps, features)")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train each model
        for model_name, model in models_dict.items():
            print(f"\nTraining: {model_name}")
            
            # Count parameters
            total_params = model.count_params()
            print(f"  Total parameters: {total_params:,}")
            
            # Train
            history = model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            epochs_trained = len(history.history['loss'])
            print(f"Completed in {epochs_trained} epochs")
            
            # Predictions
            y_train_pred = model.predict(data['X_train'], verbose=0).flatten()
            y_val_pred = model.predict(data['X_val'], verbose=0).flatten()
            y_test_pred = model.predict(data['X_test'], verbose=0).flatten()
            
            # Calculate comprehensive metrics
            train_metrics = self.calc_metrics(data['y_train'], y_train_pred)
            val_metrics = self.calc_metrics(data['y_val'], y_val_pred)
            test_metrics = self.calc_metrics(data['y_test'], y_test_pred)
            
            # Store results
            results[model_name] = {
                'model': model,
                'history': history,
                'epochs': epochs_trained,
                'params': total_params,
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred,
                    'test': y_test_pred
                }
            }
            
            # Print test metrics
            print(f"  Test Performance:")
            print(f"    MAE:              {test_metrics['mae']*100:.4f}%")
            print(f"    Huber Loss:       {test_metrics['huber_loss']*100:.4f}%")
            print(f"    Directional Acc:  {test_metrics['directional_accuracy']*100:.2f}%")
            print(f"    IC (Spearman):    {test_metrics['ic']:.4f} (p={test_metrics['ic_pvalue']:.4f})")
            print(f"    Precision (>1%):  {test_metrics['precision_large_moves']*100:.2f}%")
            print(f"    Recall (>1%):     {test_metrics['recall_large_moves']*100:.2f}%")
        
        # Create comparison table
        print("\n Test set comparison summary")
        
        comparison_data = []
        for name, res in results.items():
            comparison_data.append({
                'Model': name,
                'Params': res['params'],
                'Epochs': res['epochs'],
                'MAE%': res['test']['mae'] * 100,
                'Huber%': res['test']['huber_loss'] * 100,
                'Dir_Acc%': res['test']['directional_accuracy'] * 100,
                'IC': res['test']['ic'],
                'Prec%': res['test']['precision_large_moves'] * 100,
                'Recall%': res['test']['recall_large_moves'] * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by IC (primary metric for sequence models)
        comparison_df = comparison_df.sort_values('IC', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        
        # Overfitting check
        print("Overfitting check")
        
        for name, res in results.items():
            train_mae = res['train']['mae']
            test_mae = res['test']['mae']
            
            # Safe absolute difference (in percentage points)
            gap_abs = (test_mae - train_mae) * 100
            
            # Safe relative change
            if train_mae > 1e-6:
                gap_rel = ((test_mae - train_mae) / train_mae) * 100
                gap_rel = np.clip(gap_rel, -100, 200)  # Cap extremes
            else:
                gap_rel = 0.0
            
            # Determine status using absolute difference (more stable)
            if gap_abs > 0.5:  # 0.5 percentage points
                status = "Overfitting"
            elif gap_abs > 0.2:  # 0.2 percentage points
                status = "Slight overfitting"
            elif gap_abs < -0.5:
                status = "Test regime easier (lower volatility)"
            elif gap_abs < -0.2:
                status = "Test slightly easier"
            else:
                status = "Good generalization"
    
            print(f"  {name}:")
            print(f"    Train MAE = {train_mae*100:.4f}%, Test MAE = {test_mae*100:.4f}%")
            print(f"    Gap = {gap_abs:+.4f} pct pts ({gap_rel:+.1f}%) {status}")
        
        # Visualization
        if plot:
            n_models = len(results)
            fig = plt.figure(figsize=(16, 12))
            
            model_names = list(results.keys())
            
            # Plot 1: Key Metrics Comparison
            ax1 = plt.subplot(3, 2, 1)
            x = np.arange(n_models)
            width = 0.25
            
            mae_vals = [results[m]['test']['mae'] * 100 for m in model_names]
            dir_vals = [results[m]['test']['directional_accuracy'] * 100 for m in model_names]
            ic_vals = [results[m]['test']['ic'] * 100 for m in model_names]  # Scale IC by 100 for visibility
            
            ax1.bar(x - width, mae_vals, width, label='MAE (%)', alpha=0.8, color='coral')
            ax1.bar(x, dir_vals, width, label='Dir Acc (%)', alpha=0.8, color='skyblue')
            ax1.bar(x + width, ic_vals, width, label='IC (×100)', alpha=0.8, color='lightgreen')
            
            ax1.set_xlabel('Model', fontsize=11)
            ax1.set_ylabel('Value', fontsize=11)
            ax1.set_title('Key Metrics Comparison (Test Set)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Directional Accuracy
            ax2 = plt.subplot(3, 2, 2)
            dir_acc_vals = [results[m]['test']['directional_accuracy'] * 100 for m in model_names]
            colors = ['green' if v > 52 else 'orange' if v > 50 else 'red' for v in dir_acc_vals]
            
            ax2.barh(model_names, dir_acc_vals, alpha=0.8, color=colors)
            ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
            ax2.axvline(x=52, color='orange', linestyle='--', linewidth=2, label='Acceptable (52%)')
            ax2.axvline(x=55, color='green', linestyle='--', linewidth=2, label='Excellent (55%)')
            
            ax2.set_xlabel('Directional Accuracy (%)', fontsize=11)
            ax2.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Plot 3: Precision vs Recall (Large Moves)
            ax3 = plt.subplot(3, 2, 3)
            prec_vals = [results[m]['test']['precision_large_moves'] * 100 for m in model_names]
            rec_vals = [results[m]['test']['recall_large_moves'] * 100 for m in model_names]
            
            ax3.scatter(rec_vals, prec_vals, s=200, alpha=0.6)
            for i, name in enumerate(model_names):
                ax3.annotate(name, (rec_vals[i], prec_vals[i]), fontsize=8, ha='right')
            
            ax3.set_xlabel('Recall (%) - Catches large moves', fontsize=11)
            ax3.set_ylabel('Precision (%) - Avoids false alarms', fontsize=11)
            ax3.set_title('Precision vs Recall for Large Moves (>1%)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Information Coefficient
            ax4 = plt.subplot(3, 2, 4)
            ic_vals = [results[m]['test']['ic'] for m in model_names]
            colors = ['green' if v > 0.03 else 'orange' if v > 0.01 else 'red' for v in ic_vals]
            
            ax4.barh(model_names, ic_vals, alpha=0.8, color=colors)
            ax4.axvline(x=0, color='black', linewidth=1)
            ax4.axvline(x=0.03, color='green', linestyle='--', linewidth=1, label='Good (0.03)')
            ax4.axvline(x=0.01, color='orange', linestyle='--', linewidth=1, label='Weak (0.01)')
            
            ax4.set_xlabel('Information Coefficient (Spearman)', fontsize=11)
            ax4.set_title('Information Coefficient - Rank Correlation', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Plot 5: Training History (Validation Loss)
            ax5 = plt.subplot(3, 2, 5)
            for name in model_names:
                ax5.plot(results[name]['history'].history['val_loss'], 
                        label=name, alpha=0.7, linewidth=1.5)
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('Validation Loss (MSE)', fontsize=11)
            ax5.set_title('Validation Loss During Training', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')  # Log scale for better visualization
            
            # Plot 6: MAE Comparison (Train vs Val vs Test)
            ax6 = plt.subplot(3, 2, 6)
            x_pos = np.arange(n_models)
            width = 0.25
            
            train_mae = [results[m]['train']['mae'] * 100 for m in model_names]
            val_mae = [results[m]['val']['mae'] * 100 for m in model_names]
            test_mae = [results[m]['test']['mae'] * 100 for m in model_names]
            
            ax6.bar(x_pos - width, train_mae, width, label='Train', alpha=0.8, color='blue')
            ax6.bar(x_pos, val_mae, width, label='Val', alpha=0.8, color='orange')
            ax6.bar(x_pos + width, test_mae, width, label='Test', alpha=0.8, color='red')
            
            ax6.set_xlabel('Model', fontsize=11)
            ax6.set_ylabel('MAE (%)', fontsize=11)
            ax6.set_title('MAE: Train vs Val vs Test', fontsize=12, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
        
        # Add comparison DataFrame and best model info to results
        results['comparison_df'] = comparison_df
        results['best_by_ic'] = comparison_df.iloc[0]['Model']
        results['best_by_mae'] = comparison_df.sort_values('MAE%').iloc[0]['Model']
        results['best_by_dir_acc'] = comparison_df.sort_values('Dir_Acc%', ascending=False).iloc[0]['Model']

        return results


    def analyze_model_errors(self, model, data, set_names=['val', 'test']):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import stats
        """
        Comprehensive error analysis for sequence models, calculating statistics
        and generating visualizations for multiple datasets within a single function.
        
        Plots the 6-panel visualization for both the Validation Set and the Test Set.
        
        Parameters:
        model : keras.Model - Trained sequence model.
        data : dict - Dictionary containing 'X_set' and 'y_set' arrays.
        set_names : list - List of dataset names ('train', 'val', 'test') to analyze.
        """
        
        all_stats = {}
        data_map = {
            'train': (data.get('X_train', np.array([])), data.get('y_train', np.array([])), "Training Set"),
            'val': (data.get('X_val', np.array([])), data.get('y_val', np.array([])), "Validation Set"),
            'test': (data.get('X_test', np.array([])), data.get('y_test', np.array([])), "Test Set"),
        }

        # 1. Loop through sets, calculate metrics, and store results
        for set_name in set_names:
            if set_name not in data_map:
                print(f"Skipping unknown set_name: {set_name}")
                continue

            X, y_true, title_prefix = data_map[set_name]

            if len(y_true) == 0:
                print(f"Warning: {title_prefix} is empty. Skipping.")
                continue

            # 1a. Predictions and Residuals
            y_pred = model.predict(X, verbose=0).flatten()
            residuals = y_true - y_pred
            # Check for single-sample case before using array slicing/corrcoef
            N = len(residuals)
            
            # 1b. Basic Statistics
            abs_residuals = np.abs(residuals)
            mean_error = residuals.mean()
            std_error = residuals.std()
            mae = abs_residuals.mean()
            rmse = np.sqrt((residuals**2).mean())
            
            # 1c. Statistical Tests
            jb_stat, jb_p = stats.jarque_bera(residuals)
            _, t_p = stats.ttest_1samp(residuals, 0)
            
            # 1d. Temporal Analysis & Directional Accuracy
            error_autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if N > 1 else 0
            dir_acc = (np.sign(y_pred) == np.sign(y_true)).sum() / N
            correlation = np.corrcoef(y_true, y_pred)[0, 1]

            # 1e. Error by Magnitude Bins
            abs_returns = np.abs(y_true) * 100
            max_ret = abs_returns.max()
            bins_edges = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, max_ret + 0.01]
            bins_edges = [e for e in bins_edges if e <= max_ret + 0.01]
            if len(bins_edges) < 2:
                bins_edges = [0, max_ret + 0.01]

            bin_errors_df = pd.DataFrame({
                'abs_return': abs_returns,
                'abs_error': abs_residuals * 100
            })
            bin_errors_df['bin'] = pd.cut(bin_errors_df['abs_return'], bins=bins_edges, include_lowest=True, right=True)
            bin_means = bin_errors_df.groupby('bin', observed=True)['abs_error'].mean().fillna(0)

            # Store results
            all_stats[set_name] = {
                'title_prefix': title_prefix,
                'y_true': y_true,
                'y_pred': y_pred,
                'residuals': residuals,
                'abs_residuals': abs_residuals,
                'Mean Error (Bias)': mean_error,
                'Std of Errors': std_error,
                'MAE': mae,
                'RMSE': rmse,
                'Median Abs Error': np.median(abs_residuals),
                '95th Percentile Error': np.percentile(abs_residuals, 95),
                'JB p-value': jb_p,
                'Bias t-test p-value': t_p,
                'Lag 1 AutoCorr': error_autocorr_lag1,
                'Directional Accuracy': dir_acc,
                'Correlation (R)': correlation,
                'Bin Errors (MAE)': bin_means.to_dict()
            }

        if not all_stats:
            print("No valid datasets found for analysis.")
            return {}
            

        # 2. Display Side-by-Side Statistics
        metric_names = [
            'Mean Error (Bias)', 'Std of Errors', 'MAE', 'RMSE', 
            'Median Abs Error', '95th Percentile Error', 'Correlation (R)',
            'Directional Accuracy', 'JB p-value', 'Bias t-test p-value', 
            'Lag 1 AutoCorr'
        ]
        
        comparison_data = {
            stats['title_prefix']: {metric: stats[metric] for metric in metric_names}
            for stats in all_stats.values()
        }

        comparison_df = pd.DataFrame(comparison_data)

        for metric in ['Mean Error (Bias)', 'Std of Errors', 'MAE', 'RMSE', 
                    'Median Abs Error', '95th Percentile Error', 'Directional Accuracy']:
            comparison_df.loc[metric, :] = comparison_df.loc[metric, :] * 100
            new_metric_name = f'{metric} (%)'
            comparison_df = comparison_df.rename(index={metric: new_metric_name})

        formatted_df = comparison_df.apply(lambda x: x.map('{:.4f}'.format))
        

        print(f"Model Error Statistics Comparison: {model.name.upper()}")
        print(formatted_df)
        
        # Display Side-by-Side Bin Errors
        bin_error_data = {
            stats['title_prefix']: stats['Bin Errors (MAE)'] 
            for stats in all_stats.values() if stats.get('Bin Errors (MAE)')
        }
        
        if bin_error_data:
            bin_error_df = pd.DataFrame(bin_error_data)
            bin_error_df.index.name = '|Actual Return| Bin (%)'
            bin_error_df = bin_error_df.apply(lambda x: x.map('{:.4f}%'.format))
            
            print("\nError by Return Magnitude (MAE across Bins)")
            print(bin_error_df)
            

        # 3. Visualization for Validation and Test Sets
        # Plot for 'val' and 'test' sets
        sets_to_plot = [name for name in set_names if name in all_stats][-2:]
        
        for set_key in sets_to_plot:
            current_stats = all_stats[set_key]
            title_prefix = current_stats['title_prefix']
            
            # Unpack stats for plotting
            residuals = current_stats['residuals']
            y_true = current_stats['y_true']
            y_pred = current_stats['y_pred']
            jb_p = current_stats['JB p-value']
            corr = current_stats['Correlation (R)']
            bin_means = pd.Series(current_stats['Bin Errors (MAE)'])
            
            fig = plt.figure(figsize=(16, 12))
            
            # Plot 1: Residual Distribution
            ax1 = plt.subplot(3, 2, 1)
            ax1.hist(residuals * 100, bins=50, edgecolor='black', alpha=0.7, density=True, color='steelblue')
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax1.plot(x * 100, stats.norm.pdf(x, mu, sigma) / 100, 'r-', linewidth=2, label='Normal Distribution')
            ax1.set_xlabel('Residual (%)')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Residual Distribution - {title_prefix}', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            textstr = f'Mean: {mu*100:.4f}%\nStd: {sigma*100:.4f}%\nJB p-val: {jb_p:.4f}'
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 2: Predictions vs Actuals (Scatter)
            ax2 = plt.subplot(3, 2, 2)
            ax2.scatter(y_true * 100, y_pred * 100, alpha=0.5, s=20, color='darkblue')
            lim_min = min(y_true.min(), y_pred.min()) * 100
            lim_max = max(y_true.max(), y_pred.max()) * 100
            ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect Prediction')
            z = np.polyfit(y_true * 100, y_pred * 100, 1)
            p = np.poly1d(z)
            ax2.plot([lim_min, lim_max], p([lim_min, lim_max]), 'g-', linewidth=2, alpha=0.7, label='Actual Fit')
            ax2.set_xlabel('Actual Return (%)')
            ax2.set_ylabel('Predicted Return (%)')
            ax2.set_title(f'Predictions vs Actuals - {title_prefix}', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            # Plot 3: Errors Over Time
            ax3 = plt.subplot(3, 2, 3)
            indices = np.arange(len(residuals))
            ax3.plot(indices, residuals * 100, linewidth=0.8, alpha=0.7, color='purple')
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
            if len(residuals) >= 20:
                rolling_error = pd.Series(residuals * 100).rolling(window=20).mean()
                ax3.plot(indices, rolling_error, color='orange', linewidth=2, label='20-period MA', alpha=0.8)
                ax3.legend()
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Residual (%)')
            ax3.set_title(f'Errors Over Time - {title_prefix}', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Q-Q Plot
            ax4 = plt.subplot(3, 2, 4)
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title(f'Q-Q Plot - {title_prefix}', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Error by Return Magnitude
            ax5 = plt.subplot(3, 2, 5)
            x_pos = np.arange(len(bin_means))
            ax5.bar(x_pos, bin_means.values, alpha=0.7, color='coral', edgecolor='black') # type: ignore
            bin_labels = [f'{interval.left:.1f}-{interval.right:.1f}' if pd.notna(interval) else 'NaN' for interval in bin_means.index]
            ax5.set_xlabel('|Actual Return| Bin (%)')
            ax5.set_ylabel('Mean Absolute Error (%)')
            ax5.set_title(f'Error by Return Magnitude - {title_prefix}', fontweight='bold')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Plot 6: Residual Autocorrelation (ACF Plot)
            ax6 = plt.subplot(3, 2, 6)
            max_lag = min(20, len(residuals) - 1)
            lags = range(1, max_lag + 1)
            autocorr = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] for lag in lags]
            ax6.bar(lags, autocorr, alpha=0.7, color='teal', edgecolor='black')
            ax6.axhline(y=0, color='black', linewidth=1)
            conf_interval = 1.96 / np.sqrt(len(residuals))
            ax6.axhline(y=conf_interval, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
            ax6.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax6.set_xlabel('Lag')
            ax6.set_ylabel('Autocorrelation')
            ax6.set_title(f'Residual Autocorrelation - {title_prefix}', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
        
        return all_stats