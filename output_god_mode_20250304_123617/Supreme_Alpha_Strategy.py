
import sys
import os
sys.path.append('/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/mathematricks')

import numpy as np
import pandas as pd
from mathematricks import Strategy

class GeneratedStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "Supreme Alpha Strategy"
        self.description = "Volatility breakout strategy optimized for high-volatility market conditions"
        
    def initialize(self):
        # Add indicators
        # Unknown indicator type: atr
        # Unknown indicator type: bollinger
        self.add_rsi(period=7)
        
    def on_bar(self, bar):
        # Strategy logic
        if bar.atr > bar.atr[-5] * 1.5:
            decision = 'buy'
        elif bar.close > bar.upper_band:
            decision = 'buy'
        elif bar.rsi < 30:
            decision = 'buy'
        else:
            decision = 'hold'
        
        # Return decision
        return decision
