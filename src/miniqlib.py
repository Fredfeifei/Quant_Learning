"""
极简量化库 - 基础工具函数
用于量化入门教学，无外部依赖，函数式设计
"""

import numpy as np
import pandas as pd
from typing import Dict
import os


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    if not os.path.exists(path):
        os.makedirs(path)


def shift_exec(signal: pd.Series, n: int = 1) -> pd.Series:
    """交易在下一期执行：返回 signal.shift(n).fillna(0)。"""
    return signal.shift(n).fillna(0)


def turnover(pos: pd.Series) -> pd.Series:
    """换手率：pos.diff().abs().fillna(0)。"""
    return pos.diff().abs().fillna(0)


def apply_cost(turnover: pd.Series, commission_bps=5, slippage_bps=2, half_spread_bps=5) -> pd.Series:
    """按基点合计成本并换算为比例：单边总成本(bps)/1e4 * turnover。"""
    total_cost_bps = commission_bps + slippage_bps + half_spread_bps
    return turnover * (total_cost_bps / 1e4)


def equity(ret: pd.Series) -> pd.Series:
    """净值曲线：(1+ret).cumprod()。"""
    return (1 + ret).cumprod()


def max_drawdown(equity_series: pd.Series) -> float:
    """最大回撤：min(equity/ equity.cummax() - 1)。"""
    return (equity_series / equity_series.cummax() - 1).min()


def perf(ret: pd.Series, periods: int = 252) -> Dict[str, float]:
    """返回 {'CAGR','Vol','Sharpe','MaxDD'}（std 用 ddof=0）。"""
    eq = equity(ret)
    total_periods = len(ret)
    years = total_periods / periods
    
    cagr = (eq.iloc[-1] ** (1/years)) - 1 if years > 0 else 0
    vol = ret.std(ddof=0) * np.sqrt(periods)
    sharpe = (ret.mean() * periods) / (ret.std(ddof=0) * np.sqrt(periods)) if ret.std(ddof=0) > 0 else 0
    max_dd = max_drawdown(eq)
    
    return {
        'CAGR': cagr,
        'Vol': vol, 
        'Sharpe': sharpe,
        'MaxDD': max_dd
    }
