"""
Market clock — generates event timestamps respecting NYSE trading hours.
"""
from datetime import datetime, time, timedelta
import pandas as pd


NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)


class MarketClock:
    def __init__(self, start: str, freq: str = "1min"):
        self.start = pd.Timestamp(start, tz="America/New_York")
        self.freq = freq

    def timestamps(self, n_steps: int):
        current = self.start
        count = 0
        while count < n_steps:
            if self._is_trading(current):
                yield current.timestamp()
                count += 1
            current += pd.Timedelta(self.freq)

    @staticmethod
    def _is_trading(ts: pd.Timestamp) -> bool:
        if ts.weekday() >= 5:
            return False
        t = ts.time()
        return NYSE_OPEN <= t < NYSE_CLOSE
