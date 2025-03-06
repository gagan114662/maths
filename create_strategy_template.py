#!/usr/bin/env python3
"""
Script to create a template for a new strategy in the vault directory
"""
import os
import sys
import argparse
from pathlib import Path

STRATEGY_TEMPLATE = """\"\"\"
{strategy_name} - {strategy_description}
\"\"\"

from vault.base_strategy import BaseStrategy, Signal, Order
import numpy as np
import pandas as pd

class Strategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__()
        self.strategy_name = '{strategy_id}'
        self.granularity = "1d"
        self.stop_loss_pct = 0.15  # 15% stoploss
        self.exit_order_type = "stoploss_pct" # sl_pct , sl_abs
        self.timeInForce = "DAY"    # DAY, Expiry, IoC (immediate or cancel) , TTL (Order validity in minutes)
        self.orderQuantity = 10
        self.orderType = "MARKET"
        # Strategy-specific parameters
{parameters}
        self.data_inputs, self.tickers = self.datafeeder_inputs()
        
    def get_name(self):
        return self.strategy_name
        
    def datafeeder_inputs(self):
        # Get tickers for strategy
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 
                   'JPM', 'V', 'PG', 'UNH', 'HD', 'CRM', 'NKE', 'XOM']
        # Define required data and lookback period
        lookback = {lookback}
        data_inputs = {{'1d': {{'columns': ['open', 'high', 'close', 'low', 'volume'] , 'lookback': lookback}}}}
        return data_inputs, tickers
    
    def generate_signals(self, next_rows, market_data_df, system_timestamp, open_signals=None):
        \"\"\"
        {entry_logic}
        
        {exit_logic}
        \"\"\"
        signals = []
        return_type = None
        open_signals = open_signals or []

        for symbol in set(market_data_df["open"].columns):
            if self.granularity not in market_data_df.index.levels[0] or len(market_data_df.loc[self.granularity]) <= self.data_inputs[self.granularity]['lookback']:
                continue
                
            # Get asset data
            asset_data_df = market_data_df.loc[self.granularity].xs(symbol, axis=1, level='symbol').reset_index()
            
            # Calculate strategy-specific indicators
{indicator_calculations}
            
            # Current price
            current_price = asset_data_df.iloc[-1]['close']
            
            # Signal generation logic
            signal_strength = 0
            orderDirection = None
            
{signal_logic}
                
            if signal_strength > 0:
                # Check if we have an open signal for this symbol
                existing_signal = None
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.status == 'closed' and order.entryOrderBool:
                                existing_signal = signal
                                break
                        if existing_signal:
                            break
                
                # If we have an existing position and get a reverse signal, add exit order
                if existing_signal:
                    existing_entry = None
                    for order in existing_signal.orders:
                        if order.symbol == symbol and order.entryOrderBool and order.status == 'closed':
                            existing_entry = order
                            break
                    
                    if existing_entry and existing_entry.orderDirection != orderDirection:
                        # Cancel any existing stoploss orders
                        for order in existing_signal.orders:
                            if order.order_type == "STOPLOSS" and order.status == "open":
                                order.status = "cancel"
                                order.message = "Cancelled due to exit signal"
                                order.fresh_update = True
                        
                        # Add market exit order
                        exit_order = Order(
                            symbol=symbol,
                            orderQuantity=self.orderQuantity,
                            orderDirection=orderDirection,
                            order_type=self.orderType,
                            symbol_ltp={{system_timestamp: current_price}},
                            timeInForce=self.timeInForce,
                            entryOrderBool=False,
                            status="pending"
                        )
                        existing_signal.orders.append(exit_order)
                        existing_signal.signal_update = True
                        signals.append(existing_signal)
                        return_type = 'signals'
                
                # If no existing position and we get a signal, create new entry
                elif not existing_signal:
                    # Create Order object
                    order = Order(
                        symbol=symbol,
                        orderQuantity=self.orderQuantity,
                        orderDirection=orderDirection,
                        order_type=self.orderType,
                        symbol_ltp={{system_timestamp: current_price}},
                        timeInForce=self.timeInForce,
                        entryOrderBool=True,
                        status="pending"
                    )

                    # Create stoploss order
                    stoploss_price = current_price * (1 - self.stop_loss_pct) if orderDirection == "BUY" else current_price * (1 + self.stop_loss_pct)
                    stoploss_order = Order(
                        symbol=symbol,
                        orderQuantity=self.orderQuantity,
                        orderDirection="SELL" if orderDirection == "BUY" else "BUY",
                        order_type="STOPLOSS",
                        price=stoploss_price,
                        symbol_ltp={{system_timestamp: current_price}},
                        timeInForce=self.timeInForce,
                        entryOrderBool=False,
                        status="pending"
                    )
                    
                    # Create Signal object
                    signal = Signal(
                        strategy_name=self.strategy_name,
                        timestamp=system_timestamp,
                        orders=[order, stoploss_order],
                        signal_strength=signal_strength,
                        granularity=self.granularity,
                        signal_type="BUY_SELL",
                        market_neutral=False
                    )
                    signals.append(signal)
                    self.logger.info(f'SIGNAL GENERATED: {{symbol}}, Direction: {{orderDirection}}, Strength: {{signal_strength:.2f}}')
                    return_type = 'signals'
                    
        return return_type, signals, self.tickers
"""

def main():
    parser = argparse.ArgumentParser(description='Create a new strategy template file')
    parser.add_argument('--number', type=int, help='Strategy number (e.g. 4 for strategy_4)')
    parser.add_argument('--type', type=str, help='Strategy type (e.g. momentum, mean_reversion, volatility)')
    parser.add_argument('--name', type=str, help='Strategy name')
    parser.add_argument('--description', type=str, help='Strategy description')
    
    args = parser.parse_args()
    
    # Find existing strategy numbers if not specified
    if not args.number:
        existing_numbers = []
        vault_dir = Path("mathematricks/vault")
        for file in vault_dir.glob("strategy_*.py"):
            try:
                number = int(file.stem.split("_")[1])
                existing_numbers.append(number)
            except (IndexError, ValueError):
                continue
        
        next_number = max(existing_numbers) + 1 if existing_numbers else 4
        args.number = next_number
    
    strategy_id = f"strategy_{args.number}"
    
    # Set defaults if not specified
    if not args.type:
        args.type = "adaptive"
    
    if not args.name:
        args.name = f"Adaptive Strategy {args.number}"
    
    if not args.description:
        args.description = f"A strategy that adapts to changing market conditions using multiple indicators"
    
    # Prepare strategy template parameters
    if args.type == "mean_reversion":
        parameters = """        self.lookback_period = 20  # days for calculating mean and std deviation
        self.deviation_threshold = 2.0  # std deviations
        self.exit_threshold = 0.5  # std deviations
        self.volume_filter = True  # use volume as confirmation"""
        
        lookback = "self.lookback_period + 10"
        
        indicator_calculations = """            # Calculate mean and standard deviation
            asset_data_df['returns'] = asset_data_df['close'].pct_change()
            asset_data_df['mean'] = asset_data_df['close'].rolling(window=self.lookback_period).mean()
            asset_data_df['std'] = asset_data_df['close'].rolling(window=self.lookback_period).std()
            
            # Calculate z-score (deviation from mean in terms of standard deviations)
            asset_data_df['z_score'] = (asset_data_df['close'] - asset_data_df['mean']) / asset_data_df['std']
            
            # Volume filter
            asset_data_df['volume_sma'] = asset_data_df['volume'].rolling(window=20).mean()
            volume_ratio = asset_data_df.iloc[-1]['volume'] / asset_data_df.iloc[-1]['volume_sma']"""
        
        signal_logic = """            # Check for mean reversion signals
            z_score = asset_data_df.iloc[-1]['z_score']
            
            # Short signal when price is too high above mean
            if z_score > self.deviation_threshold and (not self.volume_filter or volume_ratio > 1.0):
                signal_strength = abs(z_score) / (self.deviation_threshold * 2)
                orderDirection = "SELL"
                
            # Long signal when price is too low below mean
            elif z_score < -self.deviation_threshold and (not self.volume_filter or volume_ratio > 1.0):
                signal_strength = abs(z_score) / (self.deviation_threshold * 2)
                orderDirection = "BUY"
                
            # Exit signals when price reverts to mean
            elif (z_score > 0 and z_score < self.exit_threshold) or (z_score < 0 and z_score > -self.exit_threshold):
                # For exit signals, we need to determine direction based on existing position
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.entryOrderBool and order.status == 'closed':
                                signal_strength = 0.5
                                orderDirection = "SELL" if order.orderDirection == "BUY" else "BUY"
                                break"""
                
        entry_logic = "Enter positions when asset price deviates significantly from its historical mean"
        exit_logic = "Exit when price reverts to the mean or crosses specified thresholds"
    
    elif args.type == "momentum":
        parameters = """        self.short_window = 20  # days for fast moving average
        self.long_window = 50  # days for slow moving average
        self.rsi_period = 14  # days for RSI calculation
        self.rsi_oversold = 30  # oversold threshold
        self.rsi_overbought = 70  # overbought threshold"""
        
        lookback = "max(self.short_window, self.long_window, self.rsi_period) + 10"
        
        indicator_calculations = """            # Calculate moving averages
            asset_data_df['short_ma'] = asset_data_df['close'].rolling(window=self.short_window).mean()
            asset_data_df['long_ma'] = asset_data_df['close'].rolling(window=self.long_window).mean()
            
            # Calculate RSI
            delta = asset_data_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            asset_data_df['rsi'] = 100 - (100 / (1 + rs))"""
        
        signal_logic = """            # Check for momentum signals with RSI filter
            rsi = asset_data_df.iloc[-1]['rsi']
            
            # Buy signal when fast MA crosses above slow MA and not overbought
            if (asset_data_df.iloc[-1]['short_ma'] > asset_data_df.iloc[-1]['long_ma']) and \
               (asset_data_df.iloc[-2]['short_ma'] <= asset_data_df.iloc[-2]['long_ma']) and \
               (rsi < self.rsi_overbought):
                signal_strength = 1.0
                orderDirection = "BUY"
                
            # Sell signal when fast MA crosses below slow MA and not oversold
            elif (asset_data_df.iloc[-1]['short_ma'] < asset_data_df.iloc[-1]['long_ma']) and \
                 (asset_data_df.iloc[-2]['short_ma'] >= asset_data_df.iloc[-2]['long_ma']) and \
                 (rsi > self.rsi_oversold):
                signal_strength = 1.0
                orderDirection = "SELL"
                
            # Exit buy positions when RSI becomes overbought
            elif rsi > self.rsi_overbought:
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.entryOrderBool and order.status == 'closed' and order.orderDirection == "BUY":
                                signal_strength = 0.8
                                orderDirection = "SELL"
                                break
                                
            # Exit sell positions when RSI becomes oversold
            elif rsi < self.rsi_oversold:
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.entryOrderBool and order.status == 'closed' and order.orderDirection == "SELL":
                                signal_strength = 0.8
                                orderDirection = "BUY"
                                break"""
                
        entry_logic = "Enter positions based on moving average crossovers filtered by RSI"
        exit_logic = "Exit when moving averages reverse or RSI reaches extreme levels"
    
    elif args.type == "volatility":
        parameters = """        self.volatility_window = 20  # days for volatility calculation
        self.volatility_z_threshold = 1.5  # standard deviations for elevated volatility
        self.atr_period = 14  # days for ATR calculation
        self.stop_loss_atr_multiple = 2.0  # ATR multiples for stop loss"""
        
        lookback = "max(self.volatility_window + 30, self.atr_period + 10)"
        
        indicator_calculations = """            # Calculate volatility metrics
            asset_data_df['returns'] = asset_data_df['close'].pct_change()
            asset_data_df['volatility'] = asset_data_df['returns'].rolling(window=self.volatility_window).std()
            asset_data_df['volatility_sma'] = asset_data_df['volatility'].rolling(window=50).mean()
            asset_data_df['volatility_z'] = (asset_data_df['volatility'] - asset_data_df['volatility_sma']) / asset_data_df['volatility'].rolling(window=100).std()
            
            # Calculate ATR (Average True Range)
            high_low = asset_data_df['high'] - asset_data_df['low']
            high_close = abs(asset_data_df['high'] - asset_data_df['close'].shift())
            low_close = abs(asset_data_df['low'] - asset_data_df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            asset_data_df['atr'] = true_range.rolling(window=self.atr_period).mean()"""
        
        signal_logic = """            # Check for volatility breakout signals
            vol_z = asset_data_df.iloc[-1]['volatility_z']
            volatility_increasing = asset_data_df.iloc[-1]['volatility'] > asset_data_df.iloc[-2]['volatility']
            
            # Volatility breakout strategy - enter when volatility spikes
            if vol_z > self.volatility_z_threshold and volatility_increasing:
                # Use recent price action to determine direction
                recent_trend = asset_data_df.iloc[-5:]['close'].pct_change().mean()
                
                if recent_trend > 0:
                    signal_strength = min(1.0, vol_z / (self.volatility_z_threshold * 2))
                    orderDirection = "BUY"
                elif recent_trend < 0:
                    signal_strength = min(1.0, vol_z / (self.volatility_z_threshold * 2))
                    orderDirection = "SELL"
                    
            # Exit when volatility normalizes
            elif vol_z < 0.5 and not volatility_increasing:
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.entryOrderBool and order.status == 'closed':
                                signal_strength = 0.7
                                orderDirection = "SELL" if order.orderDirection == "BUY" else "BUY"
                                break"""
                
        entry_logic = "Enter positions when volatility spikes significantly above its historical average"
        exit_logic = "Exit when volatility returns to normal levels or using ATR-based stops"
    
    else:  # Default to adaptive indicators
        parameters = """        self.ema_short = 10  # days for short EMA
        self.ema_medium = 30  # days for medium EMA
        self.ema_long = 50  # days for long EMA
        self.rsi_period = 14  # days for RSI
        self.volatility_lookback = 20  # days for volatility calculation
        self.sensitivity = 0.2  # sensitivity factor for parameter adjustments"""
        
        lookback = "max(self.ema_short, self.ema_medium, self.ema_long, self.rsi_period, self.volatility_lookback) + 20"
        
        indicator_calculations = """            # Calculate base indicators
            asset_data_df['returns'] = asset_data_df['close'].pct_change()
            asset_data_df['volatility'] = asset_data_df['returns'].rolling(window=self.volatility_lookback).std()
            
            # Adjust parameters based on recent volatility
            vol_ratio = asset_data_df.iloc[-1]['volatility'] / asset_data_df['volatility'].rolling(window=60).mean().iloc[-1]
            vol_adjustment = 1.0 + (vol_ratio - 1.0) * self.sensitivity
            
            # Adjust periods based on volatility
            ema_short_adj = max(5, int(self.ema_short / vol_adjustment))
            ema_medium_adj = max(15, int(self.ema_medium / vol_adjustment))
            ema_long_adj = max(30, int(self.ema_long / vol_adjustment))
            rsi_period_adj = max(7, int(self.rsi_period / vol_adjustment))
            
            # Calculate adaptive indicators
            asset_data_df['ema_short'] = asset_data_df['close'].ewm(span=ema_short_adj, adjust=False).mean()
            asset_data_df['ema_medium'] = asset_data_df['close'].ewm(span=ema_medium_adj, adjust=False).mean()
            asset_data_df['ema_long'] = asset_data_df['close'].ewm(span=ema_long_adj, adjust=False).mean()
            
            # Calculate adaptive RSI
            delta = asset_data_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period_adj).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period_adj).mean()
            rs = gain / loss
            asset_data_df['rsi'] = 100 - (100 / (1 + rs))"""
        
        signal_logic = """            # Check for adaptive signals
            ema_short = asset_data_df.iloc[-1]['ema_short']
            ema_medium = asset_data_df.iloc[-1]['ema_medium']
            ema_long = asset_data_df.iloc[-1]['ema_long']
            rsi = asset_data_df.iloc[-1]['rsi']
            
            # Bullish signals
            bullish_ema = (ema_short > ema_medium) and (ema_medium > ema_long)
            bullish_rsi = rsi > 50 and rsi < 70
            
            # Bearish signals
            bearish_ema = (ema_short < ema_medium) and (ema_medium < ema_long)
            bearish_rsi = rsi < 50 and rsi > 30
            
            # Generate entry signals
            if bullish_ema and bullish_rsi:
                signal_strength = 0.8
                orderDirection = "BUY"
            elif bearish_ema and bearish_rsi:
                signal_strength = 0.8
                orderDirection = "SELL"
                
            # Generate exit signals
            elif (not bullish_ema and orderDirection == "BUY") or (not bearish_ema and orderDirection == "SELL"):
                for signal in open_signals:
                    if signal.status not in ['closed', 'rejected']:
                        for order in signal.orders:
                            if order.symbol == symbol and order.entryOrderBool and order.status == 'closed':
                                signal_strength = 0.6
                                orderDirection = "SELL" if order.orderDirection == "BUY" else "BUY"
                                break"""
                
        entry_logic = "Enter positions based on adaptive indicators that adjust to changing market conditions"
        exit_logic = "Exit when the adaptive indicators signal a trend reversal or reach extreme levels"
    
    # Create the strategy file
    output_file = f"mathematricks/vault/{strategy_id}.py"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format the template
    content = STRATEGY_TEMPLATE.format(
        strategy_id=strategy_id,
        strategy_name=args.name,
        strategy_description=args.description,
        parameters=parameters,
        lookback=lookback,
        indicator_calculations=indicator_calculations,
        signal_logic=signal_logic,
        entry_logic=entry_logic,
        exit_logic=exit_logic
    )
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Strategy template created: {output_file}")

if __name__ == "__main__":
    main()