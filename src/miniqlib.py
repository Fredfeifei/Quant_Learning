"""
极简量化库 - 基础工具函数
用于量化入门教学，无外部依赖，函数式设计
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict


# 基础工具
def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    if not os.path.exists(path):
        os.makedirs(path)


def fetch_price(ticker: str, start: str = "2015-01-01", end: str = None, auto_adjust: bool = True) -> pd.DataFrame:
    """用 yfinance 获取 OHLCV，DatetimeIndex，列含 Open/High/Low/Close/Volume。失败时返回空 DataFrame。"""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
        if data.empty:
            print(f"Warning: No data downloaded for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


# 收益与绩效
def simple_return(px: pd.Series) -> pd.Series:
    """简单收益率：px.pct_change().fillna(0)。"""
    return px.pct_change().fillna(0)


def log_return(px: pd.Series) -> pd.Series:
    """对数收益率：np.log(px/px.shift(1)).fillna(0)。"""
    return np.log(px/px.shift(1)).fillna(0)


def equity(ret: pd.Series) -> pd.Series:
    """净值：(1+ret).cumprod()。"""
    return (1 + ret).cumprod()


def drawdown(equity_series: pd.Series) -> pd.Series:
    """回撤序列：equity/equity.cummax() - 1。"""
    return equity_series / equity_series.cummax() - 1


def max_drawdown(equity_series: pd.Series) -> float:
    """最大回撤：drawdown().min()。"""
    return drawdown(equity_series).min()


def perf(ret: pd.Series, periods: int = 252) -> Dict[str, float]:
    """返回 {'CAGR','Vol','Sharpe','MaxDD'}；std 用 ddof=0。"""
    eq = equity(ret)
    total_periods = len(ret)
    years = total_periods / periods
    
    cagr = (eq.iloc[-1] ** (1/years)) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
    vol = ret.std(ddof=0) * np.sqrt(periods)
    sharpe = (ret.mean() * periods) / (ret.std(ddof=0) * np.sqrt(periods)) if ret.std(ddof=0) > 0 else 0
    max_dd = max_drawdown(eq)
    
    return {
        'CAGR': cagr,
        'Vol': vol, 
        'Sharpe': sharpe,
        'MaxDD': max_dd
    }


# 常用指标（向量化）
def sma(px: pd.Series, window: int) -> pd.Series:
    """简单均线。"""
    return px.rolling(window=window).mean()


def ema(px: pd.Series, span: int) -> pd.Series:
    """指数均线。"""
    return px.ewm(span=span).mean()


def rsi(px: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标：涨跌分解 + EMA/均值实现。"""
    delta = px.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period).mean()
    avg_loss = loss.ewm(span=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bbands(px: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """布林带：返回(mid, up, dn, std)。"""
    mid = px.rolling(window=window).mean()
    std = px.rolling(window=window).std()
    up = mid + (std * num_std)
    dn = mid - (std * num_std)
    return mid, up, dn, std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR：基于 True Range 的 period 滚动均值。"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(px: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD：返回(macd, signal, hist)。"""
    ema_fast = px.ewm(span=fast).mean()
    ema_slow = px.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道：上/下/中轨。"""
    up = high.rolling(window=window).max()
    dn = low.rolling(window=window).min()
    mid = (up + dn) / 2
    return up, dn, mid


def obv(px: pd.Series, vol: pd.Series) -> pd.Series:
    """OBV：量价累积。"""
    price_change = px.diff()
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (direction * vol).cumsum()
    return obv


# 原有函数保留
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
