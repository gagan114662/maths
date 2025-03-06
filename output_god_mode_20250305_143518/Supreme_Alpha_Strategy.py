
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
        self.description = "Momentum strategy optimized for rising markets"
        
    def initialize(self):
        # Add indicators
        self.add_sma(period=20)
        self.add_sma(period=50)
        self.add_rsi(period=14)
        
    def on_bar(self, bar):
        # Strategy logic
        if bar.sma20 > bar.sma50:
            decision = 'buy'
        elif bar.close > bar.sma20:
            decision = 'buy'
        elif bar.rsi > 50:
            decision = 'buy'
        else:
            decision = 'hold'
        
        # Return decision
        return decision
