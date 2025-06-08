from abc import ABC, abstractmethod

import pandas as pd


class Decomposer(ABC):
    @abstractmethod
    def decompose(self, series: pd.Series) -> dict:
        """Decompose the time series into trend, seasonal, residual."""
        ...
