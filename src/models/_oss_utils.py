# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd

OSS_MONTHS_DEFAULT = 3

def split_oss_last_k(y: pd.Series, k: int = OSS_MONTHS_DEFAULT) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Returns (y_train, oss_idx). y must be monthly MS index.
    We hold out the last k months strictly for OSS.
    """
    y = y.sort_index()
    if len(y) <= k:
        return y.iloc[:0], y.index
    return y.iloc[:-k], y.index[-k:]

def month_start_index_like(y: pd.Series) -> pd.Series:
    """Ensure MS frequency (MonthStart) with no gaps."""
    return (
        y.asfreq("MS")
         .interpolate("linear", limit_direction="both")
    )
