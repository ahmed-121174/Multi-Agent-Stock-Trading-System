# pandas is used for data manipulation and analysis, especially for working with DataFrames
import pandas as pd

# numpy is used for numerical computations, including array operations and mathematical functions
import numpy as np

# yfinance is used to fetch historical stock market data from Yahoo Finance
import yfinance as yf

# GaussianNB is a Naive Bayes classifier from scikit-learn for classification tasks
from sklearn.naive_bayes import GaussianNB

# KMeans is a clustering algorithm from scikit-learn for unsupervised learning
from sklearn.cluster import KMeans

# StandardScaler is used to standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

# accuracy_score and classification_report are used to evaluate classification model performance
from sklearn.metrics import accuracy_score, classification_report

# matplotlib.pyplot is used for creating visualizations and plots
import matplotlib.pyplot as plt

# datetime and timedelta are used for handling dates and time related operations
from datetime import datetime, timedelta

# time is used for adding delays or measuring execution time
import time

# os is used for interacting with the operating system, such as file and directory operations
import os

# random is used for generating random numbers and performing random operations
import random

# warnings is used to suppress or handle warning messages
import warnings

# ThreadPoolExecutor is used for parallel execution of tasks in multiple threads
from concurrent.futures import ThreadPoolExecutor

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class NaiveBayesAgent:
    """Agent that predicts market movement using Naive Bayes classification"""
    
    def __init__(self, name="NB_Agent"):
        self.name = name
        self.model = GaussianNB()  # Initialize the Naive Bayes model
        self.trained = False  # Flag to check if the model is trained
    
    def preprocess_data(self, df):
        """Extract features from stock data"""
        # Calculate technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()  # 5 day Simple Moving Average
        df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20 day Simple Moving Average
        df['RSI'] = self._calculate_rsi(df['Close'], 14)  # Relative Strength Index (RSI)
        
        # Create target variable: 1 if tomorrow's price is higher, 0 otherwise
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Create additional features
        df['SMA_Ratio'] = df['SMA_5'] / df['SMA_20']  # Ratio of SMA_5 to SMA_20
        df['Price_to_SMA5'] = df['Close'] / df['SMA_5']  # Price relative to SMA_5
        df['Daily_Return'] = df['Close'].pct_change()  # Daily percentage return
        df['Volatility'] = df['Daily_Return'].rolling(window=5).std()  # 5 day rolling volatility
        
        # Drop rows with NaN values (caused by rolling calculations)
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()  # Calculate price changes
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average losses
        
        rs = gain / loss  # Relative strength
        rsi = 100 - (100 / (1 + rs))  # RSI formula
        return rsi
    
    def train(self, df):
        """Train the Naive Bayes model on historical data"""
        # Define features and target variable
        features = ['SMA_Ratio', 'Price_to_SMA5', 'RSI', 'Volatility']
        X = df[features]  # Feature matrix
        y = df['Target']  # Target variable
        
        # Train the Naive Bayes model
        self.model.fit(X, y)
        self.trained = True  # Mark the model as trained
        return self
    
    def predict(self, df):
        """Predict market movement"""
        if not self.trained:
            raise Exception("Model not trained yet")  # Ensure the model is trained before prediction
        
        # Define features for prediction
        features = ['SMA_Ratio', 'Price_to_SMA5', 'RSI', 'Volatility']
        X = df[features]  # Feature matrix
        
        # Return predictions and prediction probabilities
        return self.model.predict(X), self.model.predict_proba(X)

class ClusteringAgent:
    """Agent that clusters market conditions to identify patterns"""
    
    def __init__(self, n_clusters=3, name="Cluster_Agent"):
        # Set the agent name
        self.name = name
        # Define the number of clusters
        self.n_clusters = n_clusters
        # Initialize KMeans model with given number of clusters and random state for reproducibility
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        # Initialize a scaler to standardize features before clustering
        self.scaler = StandardScaler()
        # Flag to indicate whether the model has been trained
        self.trained = False
    
    def train(self, df):
        """Train the clustering model on historical data"""
        # Define the feature columns to be used for clustering
        features = ['SMA_Ratio', 'Price_to_SMA5', 'RSI', 'Volatility']
        # Extract the feature data from the DataFrame
        X = df[features]
        
        # Scale (standardize) the features to have zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the KMeans clustering model to the scaled data
        self.model.fit(X_scaled)
        
        # Create a copy of the original DataFrame to avoid modifying it directly
        df_copy = df.copy()
        # Assign the predicted cluster labels to a new 'Cluster' column in the DataFrame copy
        df_copy.loc[:, 'Cluster'] = self.model.labels_
        
        # Initialize dictionary to hold the success rate of each cluster
        cluster_success = {}
        # Loop over each cluster
        for cluster in range(self.n_clusters):
            # Filter rows belonging to the current cluster
            cluster_data = df_copy[df_copy['Cluster'] == cluster]
            # If the cluster has data points
            if len(cluster_data) > 0:
                # Calculate the average 'Target' value as the success rate
                success_rate = cluster_data['Target'].mean()
                # Store the success rate for this cluster
                cluster_success[cluster] = success_rate
        
        # Save the success rates for all clusters
        self.cluster_success = cluster_success
        # Mark the model as trained
        self.trained = True
        # Return the instance itself for chaining or further use
        return self
    
    def predict(self, df):
        """Classify market conditions into clusters"""
        # Raise an exception if the model has not been trained
        if not self.trained:
            raise Exception("Model not trained yet")
        
        # Define the feature columns to be used for prediction
        features = ['SMA_Ratio', 'Price_to_SMA5', 'RSI', 'Volatility']
        # Extract the feature data from the new DataFrame
        X = df[features]
        
        # Scale the new data using the same scaler fitted during training
        X_scaled = self.scaler.transform(X)
        
        # Predict the cluster labels for the new data
        clusters = self.model.predict(X_scaled)
        
        # Map each predicted cluster to its corresponding success rate (or 0.5 if unknown)
        confidence = [self.cluster_success.get(c, 0.5) for c in clusters]
        
        # Return both the cluster labels and their associated confidence scores
        return clusters, confidence

class SectorCorrelationAgent:
    """Agent that analyzes correlations between stocks in the same sector"""
    
    def __init__(self, name="Sector_Correlation_Agent"):
        # Set the name of the agent
        self.name = name
        # Initialize a dictionary to store correlation values
        self.correlations = {}
        # Initialize a dictionary to store sector-wide trends
        self.sector_trends = {}
        # Flag to indicate whether the model has been trained
        self.trained = False
    
    def train(self, sector_data):
        """Train on historical sector data
        
        Args:
            sector_data: Dictionary with keys as symbols and values as DataFrames
        """
        # Dictionary to hold daily return data for each symbol
        returns = {}
        for symbol, df in sector_data.items():
            # Calculate daily return if not already present
            if 'Daily_Return' not in df.columns:
                df['Daily_Return'] = df['Close'].pct_change()
            # Store daily return in returns dictionary
            returns[symbol] = df['Daily_Return']
        
        # Create a DataFrame with daily returns for all symbols
        return_df = pd.DataFrame({symbol: ret for symbol, ret in returns.items()})
        
        # Calculate the correlation matrix of returns
        self.correlation_matrix = return_df.corr()
        
        # Compute the average correlation of each stock with all others
        self.avg_correlations = {}
        for symbol in return_df.columns:
            # Exclude self-correlation and take mean
            self.avg_correlations[symbol] = self.correlation_matrix[symbol].drop(symbol).mean()
        
        # Calculate the 5-day trend for each stock in the sector
        self.sector_trends = {}
        for symbol, df in sector_data.items():
            # Ensure there are at least 5 days of data
            if len(df) >= 5:
                # Calculate percent change over the last 5 days
                self.sector_trends[symbol] = (df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1
        
        # Mark model as trained
        self.trained = True
        # Return the trained agent
        return self
    
    def predict(self, current_data):
        """Predict based on sector correlations
        
        Args:
            current_data: Dictionary with keys as symbols and values as DataFrames with latest data
        """
        # Raise error if model hasn't been trained yet
        if not self.trained:
            raise Exception("Model not trained yet")
        
        # Dictionaries to hold predictions and confidence scores
        predictions = {}
        confidences = {}
        
        # Loop through each symbol in current data
        for symbol, df in current_data.items():
            # Retrieve average correlation for this stock
            avg_corr = self.avg_correlations.get(symbol, 0)
            
            # Count how many stocks in the sector are trending up
            sector_up_count = sum(1 for trend in self.sector_trends.values() if trend > 0)
            # Calculate the proportion of trending-up stocks
            sector_sentiment = sector_up_count / len(self.sector_trends) if self.sector_trends else 0.5
            
            # If stock is highly correlated with the sector
            if avg_corr > 0.5:
                # Predict upward movement if sector sentiment is positive
                prediction = 1 if sector_sentiment > 0.5 else 0
                # Confidence scaled by deviation from neutral sentiment and correlation strength
                confidence = abs(sector_sentiment - 0.5) * 2 * avg_corr
            else:
                # For weakly correlated stocks, make a prediction but with neutral confidence
                prediction = 1 if sector_sentiment > 0.5 else 0
                confidence = 0.5
            
            # Store prediction and confidence for this symbol
            predictions[symbol] = prediction
            confidences[symbol] = confidence
        
        # Return predictions and corresponding confidence scores
        return predictions, confidences


class MacroEconomicAgent:
    """Agent that analyzes macro economic indicators and market sentiment"""
    
    def __init__(self, name="Macro_Economic_Agent"):
        # Set the name of the agent
        self.name = name
        # Dictionary to store computed market indicators
        self.market_indicators = {}
        # Flag to check if the model has been trained
        self.trained = False
    
    def train(self, market_index_data, vix_data=None):
        """Train on market index data and VIX if available
        
        Args:
            market_index_data: DataFrame with market index (e.g., S&P 500) data
        """
        # If market index data is provided
        if market_index_data is not None:
            # Calculate 10-day and 50-day simple moving averages
            market_index_data['SMA_10'] = market_index_data['Close'].rolling(window=10).mean()
            market_index_data['SMA_50'] = market_index_data['Close'].rolling(window=50).mean()
            
            # If we have at least 50 days of data, proceed
            if len(market_index_data) > 50:
                # Get the latest row of the index data
                last_row = market_index_data.iloc[-1]
                
                # Determine market trend based on moving averages
                self.market_indicators['Market_Trend'] = 1 if last_row['SMA_10'] > last_row['SMA_50'] else 0
                
                # Calculate market momentum as the return over the last 10 days
                momentum = (last_row['Close'] / market_index_data.iloc[-11]['Close']) - 1
                self.market_indicators['Market_Momentum'] = 1 if momentum > 0 else 0
                self.market_indicators['Momentum_Value'] = momentum  # Store the raw momentum value
        
        # If VIX (Volatility Index) data is provided
        if vix_data is not None and len(vix_data) > 0:
            # Calculate 5-day simple moving average for VIX
            vix_data['SMA_5'] = vix_data['Close'].rolling(window=5).mean()
            
            # Only proceed if enough data is available
            if len(vix_data) > 5:
                # Get the last row of VIX data
                last_vix = vix_data.iloc[-1]
                
                # Determine VIX trend — decreasing VIX is bullish, so reverse the logic
                vix_trend = 0 if last_vix['SMA_5'] > vix_data.iloc[-6]['SMA_5'] else 1
                self.market_indicators['VIX_Trend'] = vix_trend
                
                # Save the current VIX level
                self.market_indicators['VIX_Level'] = last_vix['Close']
                
                # If VIX is high (> 25), it's bearish; otherwise bullish
                self.market_indicators['VIX_Signal'] = 0 if last_vix['Close'] > 25 else 1
        
        # Mark that the agent has been trained
        self.trained = True
        # Return the trained agent
        return self
    
    def predict(self):
        """Predict market direction based on macro indicators"""
        # Ensure model is trained before prediction
        if not self.trained:
            raise Exception("Model not trained yet")
        
        # Count the number of positive binary signals (1s)
        positive_signals = sum(1 for signal in self.market_indicators.values() 
                              if isinstance(signal, (int, float)) and signal == 1)
        
        # Count the total number of binary signals (0 or 1)
        total_signals = sum(1 for signal in self.market_indicators.values() 
                           if isinstance(signal, (int, float)) and signal in [0, 1])
        
        # If we have any valid signals
        if total_signals > 0:
            # Compute the ratio of positive signals
            positive_ratio = positive_signals / total_signals
            
            # Predict upward market movement if more than 50% of signals are positive
            prediction = 1 if positive_ratio > 0.5 else 0
            
            # Confidence is how far we are from a neutral signal (0.5), scaled to [0, 1]
            confidence = abs(positive_ratio - 0.5) * 2
            
            return prediction, confidence
        else:
            # Not enough data to make a decision
            return 0.5, 0.5  # Neutral prediction and confidence

class CoordinationAgent:
    """Meta-agent that coordinates predictions from multiple agents across stocks"""
    
    def __init__(self, name="Portfolio_Coordinator"):
        self.name = name  # Name of the coordinator agent
        self.stock_weights = {}  # Dictionary to store relative importance (weights) of each stock
        self.agent_weights = {  # Initial weights for each prediction agent
            "NaiveBayes_Predictor": 0.3,
            "Market_Clusterer": 0.2,
            "Sector_Correlation_Agent": 0.2,
            "Macro_Economic_Agent": 0.3
        }
        self.stock_performance = {}  # Placeholder to track performance metrics for each stock
        self.position_limits = {  # Constraints on the portfolio's structure
            'max_positions': 5,  # Max number of simultaneous active positions
            'max_allocation': 0.25,  # Max capital allocation per single stock
            'min_allocation': 0.05   # Minimum capital allocation when entering a position
        }
    
    def update_stock_weights(self, symbols, market_caps=None):
        """Update the relative weights of stocks in the portfolio
        
        Args:
            symbols: List of stock symbols
            market_caps: Optional dictionary of market caps for proportional weighting
        """
        if market_caps:
            # Normalize market caps to determine relative weights
            total_mcap = sum(market_caps.values())  # Sum of all market caps
            self.stock_weights = {s: market_caps[s]/total_mcap for s in symbols}  # Weighted by market cap
        else:
            # If no market cap data, assign equal weights to all stocks
            self.stock_weights = {s: 1/len(symbols) for s in symbols}
        
        return self  # Return self for method chaining
    
    def update_agent_weights(self, performance_metrics=None):
        """Update weights given to each agent based on performance
        
        Args:
            performance_metrics: Optional dict with agent names as keys and performance metrics as values
        """
        if performance_metrics:
            # Normalize the performance scores to update weights
            total_perf = sum(performance_metrics.values())  # Total performance score
            if total_perf > 0:
                # Update weights proportionally
                self.agent_weights = {a: p/total_perf for a, p in performance_metrics.items()}
            
        return self  # Return self for method chaining
    
    def aggregate_predictions(self, prediction_dict):
        """Aggregate predictions from all agents for all stocks
        
        Args:
            prediction_dict: Nested dictionary with format:
                {stock_symbol: {agent_name: (prediction, confidence)}}
        
        Returns:
            Dictionary with stock symbols as keys and tuples of (decision, confidence) as values
        """
        aggregated_decisions = {}  # Store final decision for each stock
        
        for symbol in prediction_dict:
            weighted_vote = 0  # Sum of weighted prediction votes
            total_weight = 0  # Sum of weights used for normalization
            
            # Loop through agents' predictions for a stock
            for agent, (pred, conf) in prediction_dict[symbol].items():
                agent_weight = self.agent_weights.get(agent, 0.25)  # Use agent's weight or default
                weighted_vote += pred * conf * agent_weight  # Weighted vote = prediction × confidence × agent weight
                total_weight += conf * agent_weight  # Sum of weights for normalization
            
            # Calculate final decision based on normalized vote
            if total_weight > 0:
                decision_value = weighted_vote / total_weight  # Average weighted prediction
                decision = 1 if decision_value > 0.5 else 0  # Convert to binary decision
                confidence = abs(decision_value - 0.5) * 2  # Scale confidence between 0 and 1
            else:
                decision = 0  # Default decision if no valid predictions
                confidence = 0  # No confidence if no valid predictions
                
            aggregated_decisions[symbol] = (decision, confidence)  # Store final decision
        
        return aggregated_decisions  # Return aggregated predictions
    
    def optimize_portfolio(self, decisions, current_positions=None, available_capital=None):
        """Optimize portfolio allocations based on predictions
        
        Args:
            decisions: Dict with stock symbols as keys and (decision, confidence) as values
            current_positions: Dict with current positions and their values
            available_capital: Total available capital for new positions
            
        Returns:
            Dict with recommended actions: buy/sell/hold with allocations
        """
        portfolio_actions = {}  # Store final action plan for the portfolio
        
        # Sort all stocks by descending confidence
        sorted_stocks = sorted(decisions.items(), key=lambda x: x[1][1], reverse=True)
        
        # Extract stocks recommended for buying
        buy_candidates = [(s, c) for s, (d, c) in sorted_stocks if d == 1]
        
        # Extract stocks recommended for selling
        sell_candidates = [(s, c) for s, (d, c) in sorted_stocks if d == 0]
        
        max_positions = self.position_limits['max_positions']  # Retrieve max position limit
        
        # If portfolio has existing positions
        if current_positions:
            # Retain positions that still have a 'buy' decision
            positions_to_keep = [s for s, _ in buy_candidates if s in current_positions]
            
            for symbol in current_positions:
                if symbol in [s for s, _ in sell_candidates]:
                    # Sell if decision is negative
                    portfolio_actions[symbol] = {'action': 'SELL', 'allocation': 0}
                elif symbol in positions_to_keep:
                    # Hold if stock is still a buy candidate
                    portfolio_actions[symbol] = {'action': 'HOLD', 'allocation': current_positions[symbol]}
                else:
                    # Hold otherwise (neutral stance)
                    portfolio_actions[symbol] = {'action': 'HOLD', 'allocation': current_positions[symbol]}
        
        # Count how many slots are already occupied
        current_position_count = sum(1 for a in portfolio_actions.values() if a['action'] in ['HOLD'])
        slots_available = max_positions - current_position_count  # Determine free slots for new buys
        
        # Buy new stocks if there is room
        if slots_available > 0:
            # Choose top unheld buy candidates
            new_buys = [s for s, _ in buy_candidates if s not in portfolio_actions or 
                         portfolio_actions[s]['action'] == 'SELL'][:slots_available]
            
            # Allocate capital among new buys
            if available_capital and new_buys:
                base_allocation = 1 / len(new_buys)  # Equal base allocation
                
                # Apply min and max allocation limits
                for symbol in new_buys:
                    allocation = min(base_allocation, self.position_limits['max_allocation'])  # Cap at max
                    allocation = max(allocation, self.position_limits['min_allocation'])  # Floor at min
                    portfolio_actions[symbol] = {'action': 'BUY', 'allocation': allocation}  # Recommend buy
        
        return portfolio_actions  # Return the final portfolio strategy

class Environment:
    """Coordinates agents and evaluates trading performance"""
    
    def __init__(self, symbols, start_date, end_date, initial_capital=10000, use_sample_data=False):
        # Initialize the list of stock symbols to analyze
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        
        # Set the start date for the analysis
        self.start_date = start_date
        
        # Set the end date for the analysis
        self.end_date = end_date
        
        # Set the initial capital for the portfolio
        self.initial_capital = initial_capital
        
        # Track the current available capital
        self.current_capital = initial_capital
        
        # Dictionary to store agents for each stock symbol
        self.agents = {}
        
        # Dictionary to store raw stock data for each symbol
        self.data = {}
        
        # Dictionary to store processed stock data for each symbol
        self.processed_data = {}
        
        # Dictionary to track current positions in the portfolio
        self.positions = {}
        
        # List to store the history of all trades executed
        self.trade_history = []
        
        # Flag to indicate whether to use sample data instead of fetching live data
        self.use_sample_data = use_sample_data
        
        # Initialize the coordination agent to aggregate predictions from other agents
        self.coordinator = CoordinationAgent()
        
        # Initialize the sector correlation agent to analyze sector-wide trends
        self.sector_agent = SectorCorrelationAgent()
        
        # Initialize the macroeconomic agent to analyze macroeconomic indicators
        self.macro_agent = MacroEconomicAgent()
        
    def load_data(self):
        """Load stock data for all symbols"""
        print(f"Loading data for {len(self.symbols)} symbols...")  # Log the number of symbols being loaded
        
        # Load market index and VIX data for macro agent
        try:
            if not self.use_sample_data:  # Check if real data should be fetched
                buffer_start = (datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")  # Add buffer to start date for indicators
                print("Loading market index data (^GSPC)...")  # Log market index data loading
                self.market_index_data = yf.download(
                    "^GSPC", start=buffer_start, end=self.end_date, progress=False  # Download S&P 500 data
                )
                print("Loading VIX data (^VIX)...")  # Log VIX data loading
                self.vix_data = yf.download(
                    "^VIX", start=buffer_start, end=self.end_date, progress=False  # Download VIX data
                )
            else:
                print("Using sample macro data")  # Log that sample data is being used
                self._generate_sample_macro_data()  # Generate synthetic macro data
        except Exception as e:
            print(f"Error loading macro data: {str(e)}")  # Log any errors during macro data loading
            self._generate_sample_macro_data()  # Fallback to generating synthetic macro data
        
        # Load individual stock data in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(self.symbols))) as executor:  # Use a thread pool for parallel data loading
            # Submit loading tasks
            future_to_symbol = {
                executor.submit(self._load_symbol_data, symbol): symbol  # Submit a task for each symbol
                for symbol in self.symbols
            }
            
            # Process results as they complete
            for future in future_to_symbol:  # Iterate over completed futures
                symbol = future_to_symbol[future]  # Get the symbol associated with the future
                try:
                    data = future.result()  # Retrieve the result of the future
                    if data is not None and len(data) > 0:  # Check if valid data was returned
                        self.data[symbol] = data  # Store the data for the symbol
                        print(f"Successfully loaded data for {symbol} ({len(data)} records)")  # Log successful data loading
                    else:
                        print(f"No data received for {symbol}, excluding from analysis")  # Log if no data was received
                except Exception as e:
                    print(f"Error loading data for {symbol}: {str(e)}")  # Log any errors during data loading
        
        # Update the symbols list to only include those with valid data
        self.symbols = list(self.data.keys())  # Filter symbols to only those with successfully loaded data
        print(f"Successfully loaded data for {len(self.symbols)} symbols")  # Log the number of successfully loaded symbols
        
        # Initialize agent weights based on available symbols
        self.coordinator.update_stock_weights(self.symbols)  # Update stock weights in the coordination agent
        
        return self  # Return the instance for method chaining
    
    def _load_symbol_data(self, symbol):
        """Load data for a single symbol"""
        # If use_sample_data is True, skip trying to download from Yahoo Finance
        if self.use_sample_data:
            print(f"Using sample data for {symbol} as requested")
            return self._generate_synthetic_stock_data(symbol)
            
        # Try to load from Yahoo Finance
        try:
            # Add some buffer to the start date for calculating indicators
            buffer_start = (datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")
            
            # Try to download with a retry mechanism
            for attempt in range(3):
                try:
                    # Add a random delay between attempts to avoid rate limiting
                    if attempt > 0:
                        delay = random.uniform(1, 3)
                        time.sleep(delay)
                    
                    data = yf.download(
                        symbol, 
                        start=buffer_start, 
                        end=self.end_date,
                        progress=False,
                        show_errors=False,
                        timeout=20
                    )
                    
                    # Check if we got data
                    if data is not None and len(data) > 0:
                        return data
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"All download attempts failed for {symbol}")
            
            # If we get here, all attempts failed
            return self._generate_synthetic_stock_data(symbol)
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            return self._generate_synthetic_stock_data(symbol)
    
    def _generate_sample_macro_data(self):
        """Generate sample market index and VIX data"""
        print("Generating sample market and VIX data")  # Log the generation of sample data
        
        # Define the start and end dates for the sample data, adding a buffer to the start date
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=100)
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Generate a range of dates between the start and end dates
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Initialize the starting price for the market index (e.g., S&P 500-like)
        initial_price = 4000.0
        # Generate daily returns with a small positive drift and some volatility
        daily_returns = np.random.normal(0.0003, 0.01, len(date_range))
        prices = [initial_price]  # Start with the initial price
        
        # Simulate the price series using the daily returns
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create a DataFrame for the market index data
        self.market_index_data = pd.DataFrame({
            'Open': prices,  # Simulated opening prices
            'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],  # Simulated high prices
            'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],  # Simulated low prices
            'Close': prices,  # Simulated closing prices
            'Adj Close': prices,  # Adjusted closing prices (same as close here)
            'Volume': [int(np.random.normal(2000000, 500000)) for _ in prices]  # Simulated trading volume
        }, index=date_range)
        
        # Initialize the starting value for the VIX (Volatility Index)
        initial_vix = 20.0
        vix_values = []  # List to store simulated VIX values
        
        # Simulate the VIX values, which tend to move inversely to the market
        for i, ret in enumerate(daily_returns):
            # Calculate the VIX change based on the market return and some noise
            vix_change = -0.7 * ret + np.random.normal(0, 0.05)
            
            if i == 0:
                # For the first day, use the initial VIX value
                vix_values.append(initial_vix)
            else:
                # Calculate the new VIX value, ensuring it doesn't drop below a floor of 9.0
                new_vix = vix_values[-1] * (1 + vix_change)
                vix_values.append(max(9.0, new_vix))
        
        # Create a DataFrame for the VIX data
        self.vix_data = pd.DataFrame({
            'Open': vix_values,  # Simulated opening VIX values
            'High': [v * (1 + abs(np.random.normal(0, 0.02))) for v in vix_values],  # Simulated high VIX values
            'Low': [v * (1 - abs(np.random.normal(0, 0.02))) for v in vix_values],  # Simulated low VIX values
            'Close': vix_values,  # Simulated closing VIX values
            'Adj Close': vix_values,  # Adjusted closing VIX values (same as close here)
            'Volume': [int(np.random.normal(500000, 100000)) for _ in vix_values]  # Simulated trading volume
        }, index=date_range)
    
    def _generate_synthetic_stock_data(self, symbol):
        """Generate synthetic stock data for testing"""
        print(f"Generating synthetic data for {symbol}")  # Log the generation of synthetic data for the symbol
        
        # Define the start and end dates for the synthetic data, adding a buffer to the start date
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=100)
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Generate a range of dates between the start and end dates
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Use the ASCII value of the first letter of the symbol to create a seed for variety
        seed = ord(symbol[0]) % 10
        # Set the initial price based on the seed to create different starting points
        initial_price = 50.0 + seed * 20.0
        
        # Define a slight positive drift and volatility based on the seed for variety
        drift = 0.0005 + (seed * 0.0001)
        volatility = 0.015 + (seed * 0.002)
        
        # Generate daily returns using a normal distribution with the defined drift and volatility
        daily_returns = np.random.normal(drift, volatility, len(date_range))
        # Initialize the price series with the initial price
        prices = [initial_price]
        
        # Simulate the price series using the daily returns
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create a DataFrame with the simulated stock data
        synthetic_data = pd.DataFrame({
            'Open': prices,  # Simulated opening prices
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],  # Simulated high prices
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],  # Simulated low prices
            'Close': prices,  # Simulated closing prices
            'Adj Close': prices,  # Adjusted closing prices (same as close here)
            'Volume': [int(np.random.normal(1000000, 300000)) for _ in prices]  # Simulated trading volume
        }, index=date_range)
        
        # Define a directory to store the synthetic data files
        sample_dir = 'sample_data'
        # Create the directory if it doesn't already exist
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save the synthetic data to a CSV file for future use
        try:
            sample_file = os.path.join(sample_dir, f"{symbol}_sample.csv")  # Define the file path
            synthetic_data.to_csv(sample_file)  # Save the DataFrame to the file
        except Exception as e:
            # Log a warning if the file could not be saved
            print(f"Warning: Could not save synthetic data to file: {str(e)}")
            
        return synthetic_data  # Return the generated synthetic data
        
    def prepare_data(self):
        """Prepare data for all symbols"""
        if not self.data:
            raise ValueError("No data available. Call load_data() first.")  # Ensure data is loaded before processing
        
        for symbol, df in self.data.items():
            print(f"Preparing data for {symbol}...")  # Log the symbol being processed
            
            # Create a copy to avoid modifying the original data
            prepared_df = df.copy()
            
            # Calculate 5-day Simple Moving Average (SMA)
            prepared_df['SMA_5'] = prepared_df['Close'].rolling(window=5).mean()
            # Calculate 20-day Simple Moving Average (SMA)
            prepared_df['SMA_20'] = prepared_df['Close'].rolling(window=20).mean()
            # Calculate Relative Strength Index (RSI) with a 14-day window
            prepared_df['RSI'] = self._calculate_rsi(prepared_df['Close'], 14)
            # Calculate the ratio of SMA_5 to SMA_20
            prepared_df['SMA_Ratio'] = prepared_df['SMA_5'] / prepared_df['SMA_20']
            # Calculate the price relative to SMA_5
            prepared_df['Price_to_SMA5'] = prepared_df['Close'] / prepared_df['SMA_5']
            # Calculate daily percentage return
            prepared_df['Daily_Return'] = prepared_df['Close'].pct_change()
            # Calculate 5-day rolling standard deviation of daily returns (volatility)
            prepared_df['Volatility'] = prepared_df['Daily_Return'].rolling(window=5).std()
            
            # Create target variable: 1 if the next day's price is higher, 0 otherwise
            prepared_df['Target'] = (prepared_df['Close'].shift(-1) > prepared_df['Close']).astype(int)
            
            # Drop rows with NaN values caused by rolling calculations
            prepared_df.dropna(inplace=True)
            
            # Filter data to include only rows within the requested date range
            prepared_df = prepared_df[prepared_df.index >= self.start_date]
            
            # Store the processed data for the symbol
            self.processed_data[symbol] = prepared_df
            
            print(f"Prepared {len(prepared_df)} records for {symbol}")  # Log the number of records prepared
        
        return self  # Return the instance for method chaining
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()  # Calculate the difference between consecutive prices
        # Handle division by zero by replacing infinities with NaN
        delta = delta.replace([np.inf, -np.inf], np.nan)
        
        # Calculate average gains over the specified window
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        # Calculate average losses over the specified window
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero by replacing zero losses with a very small value
        loss = loss.replace(0, np.finfo(float).eps)
        
        # Calculate Relative Strength (RS)
        rs = gain / loss
        # Calculate RSI using the formula
        rsi = 100 - (100 / (1 + rs))
        return rsi  # Return the calculated RSI values
    
    def setup_agents(self):
        """Set up individual agents for each stock"""
        for symbol in self.symbols:
            # Create a Naive Bayes agent for the stock
            nb_agent = NaiveBayesAgent(name=f"NaiveBayes_Predictor")
            # Create a clustering agent with 3 clusters for the stock
            cluster_agent = ClusteringAgent(n_clusters=3, name=f"Market_Clusterer")
            
            # Store the agents for this symbol in the agents dictionary
            self.agents[symbol] = {
                "NaiveBayes_Predictor": nb_agent,
                "Market_Clusterer": cluster_agent
            }
        
        # Train the macroeconomic agent using market index and VIX data
        self.macro_agent.train(self.market_index_data, self.vix_data)
        
        # Train the sector correlation agent using processed stock data
        self.sector_agent.train(self.processed_data)
        
        return self  # Return the instance for method chaining
    
    def train_agents(self, train_size=0.7):
        """Train all individual stock agents with historical data"""
        if not self.processed_data:
            raise ValueError("No processed data available. Call prepare_data() first.")  # Ensure data is prepared before training
        
        self.train_data = {}  # Dictionary to store training data for each stock
        self.test_data = {}  # Dictionary to store testing data for each stock
        
        # Train individual stock agents
        for symbol, agents in self.agents.items():
            if symbol not in self.processed_data:
                print(f"Warning: No data for {symbol}, skipping agent training")  # Skip training if no data is available
                continue
                
            df = self.processed_data[symbol]  # Get processed data for the stock
            
            # Split data into training and testing
            train_idx = int(len(df) * train_size)  # Calculate training data size
            
            # Ensure we have some data for training and testing
            if train_idx < 20:  # Minimum data requirement for training
                train_idx = max(20, int(len(df) * 0.7))  # Adjust training size if too small
                
            # Ensure we have some data left for testing
            if train_idx >= len(df) - 5:  # Leave at least 5 samples for testing
                train_idx = max(20, len(df) - 5)  # Adjust training size if too large
                
            train_data = df.iloc[:train_idx]  # Extract training data
            test_data = df.iloc[train_idx:]  # Extract testing data
            
            print(f"Training agents for {symbol} with {len(train_data)} records, testing with {len(test_data)} records")  # Log training/testing split
            
            # Train each agent
            for agent_name, agent in agents.items():
                try:
                    agent.train(train_data)  # Train the agent with training data
                    print(f"Successfully trained {agent_name} for {symbol}")  # Log successful training
                except Exception as e:
                    print(f"Error training {agent_name} for {symbol}: {str(e)}")  # Log training errors
            
            # Store the train/test splits
            self.train_data[symbol] = train_data  # Save training data
            self.test_data[symbol] = test_data  # Save testing data
        
        return self  # Return the instance for method chaining
    
    def evaluate(self):
        """Evaluate the trading strategy on test data for the entire portfolio"""
        if not hasattr(self, 'test_data') or not self.test_data:
            raise ValueError("No test data available. Run train_agents() first.")  # Ensure test data is available
        
        # Initialize portfolio and tracking variables
        portfolio_value = self.initial_capital  # Start with initial capital
        cash = portfolio_value  # Initialize cash with the same value
        positions = {}  # Dictionary to track current positions (symbol -> {shares, value})
        trade_history = []  # List to store trade history
        daily_portfolio_values = []  # List to track daily portfolio values
        
        # Align dates across all stocks to find the common testing period
        common_dates = None  # Initialize common dates
        for symbol, test_df in self.test_data.items():
            dates = set(test_df.index)  # Get dates for the stock
            if common_dates is None:
                common_dates = dates  # Initialize common dates with the first stock's dates
            else:
                common_dates = common_dates.intersection(dates)  # Find intersection of dates
        
        common_dates = sorted(list(common_dates))  # Sort the common dates
        print(f"Evaluating strategy across {len(common_dates)} common trading days")  # Log the number of trading days
        
        if len(common_dates) < 5:
            print("Warning: Very few common trading days available for evaluation")  # Warn if too few trading days
        
        # Loop through each trading day
        for date in common_dates:
            current_data = {}  # Dictionary to store current day's data for each stock
            
            # Get current day's data for each stock
            for symbol, test_df in self.test_data.items():
                if date in test_df.index:
                    row = test_df.loc[date]  # Get the row for the current date
                    current_data[symbol] = pd.DataFrame([row])  # Store it as a DataFrame
            
            # Collect predictions from all agents for all stocks
            all_predictions = {}  # Dictionary to store predictions
            
            for symbol, data in current_data.items():
                all_predictions[symbol] = {}  # Initialize predictions for the stock
                
                # Get predictions from stock-specific agents
                for agent_name, agent in self.agents[symbol].items():
                    try:
                        if isinstance(agent, NaiveBayesAgent):  # If the agent is NaiveBayesAgent
                            pred, conf = agent.predict(data)  # Get predictions and confidence
                            all_predictions[symbol][agent_name] = (pred[0], conf[0][1])  # Store predictions
                        elif isinstance(agent, ClusteringAgent):  # If the agent is ClusteringAgent
                            cluster, conf = agent.predict(data)  # Get cluster and confidence
                            pred = 1 if conf[0] > 0.5 else 0  # Convert confidence to binary prediction
                            all_predictions[symbol][agent_name] = (pred, conf[0])  # Store predictions
                    except Exception as e:
                        print(f"Error getting prediction from {agent_name} for {symbol}: {str(e)}")  # Log prediction errors
                        all_predictions[symbol][agent_name] = (0.5, 0.5)  # Neutral prediction
            
            # Get prediction from sector correlation agent
            try:
                sector_preds, sector_confs = self.sector_agent.predict(current_data)  # Get sector predictions
                for symbol in current_data:
                    if symbol in sector_preds:
                        all_predictions[symbol]["Sector_Correlation_Agent"] = (sector_preds[symbol], sector_confs[symbol])  # Store predictions
            except Exception as e:
                print(f"Error getting sector predictions: {str(e)}")  # Log prediction errors
                # Add neutral prediction for all symbols
                for symbol in current_data:
                    all_predictions[symbol]["Sector_Correlation_Agent"] = (0.5, 0.5)
            
            # Get prediction from macro agent (same for all stocks)
            try:
                macro_pred, macro_conf = self.macro_agent.predict()  # Get macro predictions
                for symbol in current_data:
                    all_predictions[symbol]["Macro_Economic_Agent"] = (macro_pred, macro_conf)  # Store predictions
            except Exception as e:
                print(f"Error getting macro prediction: {str(e)}")  # Log prediction errors
                for symbol in current_data:
                    all_predictions[symbol]["Macro_Economic_Agent"] = (0.5, 0.5)
            
            # Get final decisions from coordination agent
            decisions = self.coordinator.aggregate_predictions(all_predictions)  # Aggregate predictions
            
            # Calculate current position values
            current_positions = {}  # Dictionary to store current position values
            for symbol, pos in positions.items():
                if symbol in current_data:
                    price = current_data[symbol]['Close'].iloc[0]  # Get current price
                    current_positions[symbol] = pos['shares'] * price  # Calculate position value
            
            # Get portfolio actions from coordinator
            actions = self.coordinator.optimize_portfolio(
                decisions, 
                current_positions=current_positions, 
                available_capital=cash
            )  # Optimize portfolio
            
            # Execute trades based on actions
            for symbol, action_info in actions.items():
                price = current_data[symbol]['Close'].iloc[0]  # Get current price
                action = action_info['action']  # Get action (BUY/SELL)
                allocation = action_info['allocation']  # Get allocation
                
                if action == 'BUY' and symbol not in positions:
                    # Calculate number of shares to buy
                    amount_to_invest = portfolio_value * allocation  # Calculate investment amount
                    shares = amount_to_invest / price  # Calculate number of shares
                    
                    # Ensure we have enough cash
                    if amount_to_invest <= cash:
                        positions[symbol] = {'shares': shares, 'entry_price': price}  # Add position
                        cash -= amount_to_invest  # Deduct cash
                        
                        trade_history.append({
                            'Date': date,
                            'Symbol': symbol,
                            'Action': 'BUY',
                            'Price': price,
                            'Shares': shares,
                            'Value': amount_to_invest
                        })  # Record trade
                
                elif action == 'SELL' and symbol in positions:
                    # Sell existing position
                    shares = positions[symbol]['shares']  # Get number of shares
                    position_value = shares * price  # Calculate position value
                    cash += position_value  # Add cash
                    
                    trade_history.append({
                        'Date': date,
                        'Symbol': symbol,
                        'Action': 'SELL',
                        'Price': price,
                        'Shares': shares,
                        'Value': position_value
                    })  # Record trade
                    
                    del positions[symbol]  # Remove position
            
            # Calculate end-of-day portfolio value
            portfolio_value = cash  # Start with cash
            for symbol, pos in positions.items():
                if symbol in current_data:
                    price = current_data[symbol]['Close'].iloc[0]  # Get current price
                    portfolio_value += pos['shares'] * price  # Add position value
            
            # Record daily portfolio value
            daily_portfolio_values.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': cash,
                'Positions': len(positions)
            })  # Save daily values
        
        # Sell all positions at the end
        final_trades = []  # List to store final trades
        for symbol, pos in positions.items():
            if symbol in self.test_data and common_dates[-1] in self.test_data[symbol].index:
                price = self.test_data[symbol].loc[common_dates[-1]]['Close']  # Get final price
                shares = pos['shares']  # Get number of shares
                position_value = shares * price  # Calculate position value
                cash += position_value  # Add cash
                
                final_trades.append({
                    'Date': common_dates[-1],
                    'Symbol': symbol,
                    'Action': 'SELL',
                    'Price': price,
                    'Shares': shares,
                    'Value': position_value
                })  # Record trade
        
        trade_history.extend(final_trades)  # Add final trades to history
        self.final_capital = cash  # Update final capital
        self.trade_history = trade_history  # Save trade history
        self.daily_portfolio_values = pd.DataFrame(daily_portfolio_values)  # Save daily values
        
        # Calculate portfolio metrics
        initial_date = common_dates[0]  # Get initial date
        final_date = common_dates[-1]  # Get final date
        
        # Calculate portfolio return
        portfolio_return = (self.final_capital / self.initial_capital) - 1  # Calculate return
        
        # Calculate benchmark returns (equal-weighted portfolio of all stocks)
        benchmark_returns = {}  # Dictionary to store benchmark returns
        for symbol, test_df in self.test_data.items():
            if initial_date in test_df.index and final_date in test_df.index:
                initial_price = test_df.loc[initial_date]['Close']  # Get initial price
                final_price = test_df.loc[final_date]['Close']  # Get final price
                benchmark_returns[symbol] = (final_price / initial_price) - 1  # Calculate return
        
        avg_benchmark_return = sum(benchmark_returns.values()) / len(benchmark_returns) if benchmark_returns else 0  # Calculate average return
        
        return {
            'Initial Capital': self.initial_capital,  # Initial capital
            'Final Capital': self.final_capital,  # Final capital
            'Portfolio Return': portfolio_return,  # Portfolio return
            'Benchmark Return': avg_benchmark_return,  # Benchmark return
            'Number of Trades': len(trade_history),  # Number of trades
            'Trade History': trade_history,  # Trade history
            'Daily Portfolio Values': self.daily_portfolio_values  # Daily portfolio values
        }  # Return evaluation results
    
    def plot_results(self):
        """Plot trading results for the portfolio"""
        # Check if daily portfolio values are available; if not, print a message and return
        if not hasattr(self, 'daily_portfolio_values') or self.daily_portfolio_values is None:
            print("No portfolio data available. Run evaluate() first.")
            return
        
        # Create a new figure for the plots with specified size
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Portfolio Value Over Time
        plt.subplot(2, 1, 1)  # Create the first subplot (2 rows, 1 column, 1st plot)
        plt.plot(self.daily_portfolio_values['Date'], self.daily_portfolio_values['Portfolio_Value'], label='Portfolio Value', linewidth=2)  # Plot portfolio value over time
        plt.title('Portfolio Performance', fontsize=14)  # Set the title of the plot
        plt.ylabel('Value ($)', fontsize=12)  # Set the y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines with transparency
        plt.legend()  # Add a legend to the plot
        
        # Plot 2: Cash and Position Count
        plt.subplot(2, 2, 3)  # Create the second subplot (2 rows, 2 columns, 3rd plot)
        plt.plot(self.daily_portfolio_values['Date'], self.daily_portfolio_values['Cash'], label='Cash', color='green')  # Plot available cash over time
        plt.title('Available Cash', fontsize=14)  # Set the title of the plot
        plt.ylabel('Cash ($)', fontsize=12)  # Set the y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines with transparency
        plt.legend()  # Add a legend to the plot
        
        plt.subplot(2, 2, 4)  # Create the fourth subplot (2 rows, 2 columns, 4th plot)
        plt.plot(self.daily_portfolio_values['Date'], self.daily_portfolio_values['Positions'], label='Active Positions', color='purple')  # Plot the number of active positions over time
        plt.title('Number of Active Positions', fontsize=14)  # Set the title of the plot
        plt.ylabel('Count', fontsize=12)  # Set the y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines with transparency
        plt.legend()  # Add a legend to the plot
        
        plt.tight_layout()  # Adjust subplot spacing to prevent overlap
        plt.show()  # Display the plots
        
        # Plot individual stock performance if the number of symbols is reasonable
        if len(self.symbols) <= 8:  # Only show individual plots for up to 8 stocks
            plt.figure(figsize=(15, 10))  # Create a new figure for individual stock plots
            cols = min(2, len(self.symbols))  # Determine the number of columns (max 2)
            rows = (len(self.symbols) + cols - 1) // cols  # Calculate the number of rows needed
            
            # Loop through each symbol to create individual plots
            for i, symbol in enumerate(self.symbols):
                if symbol in self.test_data:  # Check if test data is available for the symbol
                    plt.subplot(rows, cols, i+1)  # Create a subplot for the symbol
                    
                    test_data = self.test_data[symbol]  # Get the test data for the symbol
                    plt.plot(test_data.index, test_data['Close'], label=f'{symbol} Price')  # Plot the closing price
                    
                    # Add buy/sell markers to the plot
                    symbol_trades = [t for t in self.trade_history if t['Symbol'] == symbol]  # Filter trades for the symbol
                    buy_trades = [t for t in symbol_trades if t['Action'] == 'BUY']  # Filter buy trades
                    sell_trades = [t for t in symbol_trades if t['Action'] == 'SELL']  # Filter sell trades
                    
                    if buy_trades:  # If there are buy trades, add markers
                        buy_dates = [t['Date'] for t in buy_trades]  # Extract buy dates
                        buy_prices = [t['Price'] for t in buy_trades]  # Extract buy prices
                        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')  # Add buy markers
                    
                    if sell_trades:  # If there are sell trades, add markers
                        sell_dates = [t['Date'] for t in sell_trades]  # Extract sell dates
                        sell_prices = [t['Price'] for t in sell_trades]  # Extract sell prices
                        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')  # Add sell markers
                    
                    plt.title(f'{symbol} Trading Activity')  # Set the title of the plot
                    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines with transparency
                    if i == 0:  # Only show legend for the first plot
                        plt.legend()
            
            plt.tight_layout()  # Adjust subplot spacing to prevent overlap
            plt.show()  # Display the plots
        
        # Print a summary of portfolio performance
        print("\n===== Portfolio Performance Summary =====")
        print(f"Initial Capital: ${self.initial_capital:.2f}")  # Print the initial capital
        print(f"Final Capital: ${self.final_capital:.2f}")  # Print the final capital
        
        portfolio_return = (self.final_capital / self.initial_capital - 1) * 100  # Calculate total portfolio return
        print(f"Total Return: {portfolio_return:.2f}%")  # Print the total return
        
        # Calculate and print annualized return if there are enough trading days
        if len(self.daily_portfolio_values) > 1:  # Ensure there are multiple trading days
            first_date = self.daily_portfolio_values['Date'].iloc[0]  # Get the first trading date
            last_date = self.daily_portfolio_values['Date'].iloc[-1]  # Get the last trading date
            trading_days = len(self.daily_portfolio_values)  # Count the number of trading days
            
            if trading_days > 0:  # Ensure there are trading days to calculate annualized return
                annualized_return = ((1 + portfolio_return/100) ** (252/trading_days) - 1) * 100  # Calculate annualized return
                print(f"Annualized Return: {annualized_return:.2f}%")  # Print the annualized return
        
        print(f"Number of Trades: {len(self.trade_history)}")  # Print the total number of trades
        
        # Calculate and print basic trading statistics
        if self.trade_history:  # Ensure there is trade history to analyze
            winning_trades = 0  # Initialize the count of winning trades
            total_pl = 0  # Initialize the total profit/loss
            
            # Group trades by symbol to calculate profit/loss for each stock
            symbol_trades = {}
            for trade in self.trade_history:
                symbol = trade['Symbol']  # Get the stock symbol for the trade
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []  # Initialize the list of trades for the symbol
                symbol_trades[symbol].append(trade)  # Add the trade to the list
            
            # Calculate profit/loss for completed round trips (buy-sell pairs)
            for symbol, trades in symbol_trades.items():
                buy_trades = [t for t in trades if t['Action'] == 'BUY']  # Filter buy trades
                sell_trades = [t for t in trades if t['Action'] == 'SELL']  # Filter sell trades
                
                # Match buys and sells in sequence (FIFO)
                while buy_trades and sell_trades:
                    buy = buy_trades.pop(0)  # Get the first buy trade
                    sell = sell_trades.pop(0)  # Get the first sell trade
                    
                    buy_value = buy['Value']  # Get the value of the buy trade
                    sell_value = sell['Value']  # Get the value of the sell trade
                    trade_pl = sell_value - buy_value  # Calculate profit/loss for the trade
                    total_pl += trade_pl  # Add to the total profit/loss
                    
                    if trade_pl > 0:  # Check if the trade was profitable
                        winning_trades += 1  # Increment the count of winning trades
            
            total_completed_trades = winning_trades + (len(self.trade_history) // 2 - winning_trades)  # Calculate total completed trades
            if total_completed_trades > 0:  # Ensure there are completed trades to calculate win rate
                win_rate = (winning_trades / total_completed_trades) * 100  # Calculate win rate
                print(f"Win Rate: {win_rate:.2f}%")  # Print the win rate
                print(f"Total P&L from Closed Positions: ${total_pl:.2f}")  # Print the total profit/loss


# Run the multi-agent trading system with specified stocks
def run_multi_stock_trading_system(
    symbols=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD'],  # List of stock symbols to trade
    start_date='2022-01-01',  # Start date for the trading period
    end_date='2022-12-31',  # End date for the trading period
    initial_capital=100000,  # Initial capital for the portfolio
    use_sample_data=False  # Flag to indicate whether to use sample data
):
    """Run the multi-agent stock trading system with multiple stocks"""
    print(f"Starting multi-stock trading system with {len(symbols)} symbols")  # Log the number of symbols
    print(f"Trading period: {start_date} to {end_date}")  # Log the trading period
    print(f"Initial capital: ${initial_capital}")  # Log the initial capital
    print(f"Using sample data: {use_sample_data}")  # Log whether sample data is being used
    
    # Create environment
    env = Environment(
        symbols=symbols,  # Pass the list of stock symbols
        start_date=start_date,  # Pass the start date
        end_date=end_date,  # Pass the end date
        initial_capital=initial_capital,  # Pass the initial capital
        use_sample_data=use_sample_data  # Pass the sample data flag
    )
    
    # Load and prepare data
    env.load_data().prepare_data()  # Load stock data and prepare it for analysis
    
    # Set up and train agents
    env.setup_agents().train_agents(train_size=0.7)  # Set up agents and train them with 70% of the data
    
    # Evaluate trading strategy
    results = env.evaluate()  # Evaluate the trading strategy on the test data
    
    # Print results
    print("\n===== Trading System Results =====")  # Print a header for the results
    print(f"Trading Results for {len(symbols)} stocks from {start_date} to {end_date}")  # Log the trading period and number of stocks
    print("-" * 50)  # Print a separator line
    print(f"Initial Capital: ${results['Initial Capital']:.2f}")  # Log the initial capital
    print(f"Final Capital: ${results['Final Capital']:.2f}")  # Log the final capital
    print(f"Strategy Return: {results['Portfolio Return']*100:.2f}%")  # Log the strategy return as a percentage
    print(f"Benchmark Return: {results['Benchmark Return']*100:.2f}%")  # Log the benchmark return as a percentage
    print(f"Number of Trades: {results['Number of Trades']}")  # Log the total number of trades executed
    
    # Plot results
    env.plot_results()  # Plot the portfolio performance and trading activity
    
    return env, results  # Return the environment and results for further analysis


if __name__ == "__main__":
    # Define symbols to trade (8 stocks across different sectors)
    symbols = [
        'AAPL',  # Technology
        'MSFT',  # Technology
        'AMZN',  # Consumer Cyclical
        'GOOGL', # Communication Services
        'META',  # Communication Services
        'TSLA',  # Automotive
        'NVDA',  # Semiconductors
        'AMD'    # Semiconductors
    ]
    
    # Run the system with multiple stocks, using sample data by default
    env, results = run_multi_stock_trading_system(
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2022-12-31',
        initial_capital=100000,
        use_sample_data=True  # Set to True to skip trying yfinance and use sample data directly
    )