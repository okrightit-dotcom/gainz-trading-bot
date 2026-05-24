import os
import json
import asyncio
import logging
import threading
import tempfile
import re
import websockets
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import aiohttp
import requests
import portalocker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from huggingface_hub import HfApi

# ══════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════
#  SECRETS
# ══════════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PHANTOM_KEY      = os.getenv("PHANTOM_KEY", "")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

# ══════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════
CONFIG_PATH       = "config.json"
ACTIVE_TRADE_PATH = "active_trades.json"

# ══════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════
JUPITER_FEE_PCT   = 0.003   # 0.3% per trade
SOL_NETWORK_FEE   = 0.000005 # SOL per tx
SOL_MINT  = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BG = "#0d1117"
FG = "#c9d1d9"

# Anti-Repainting: Tracking live bars to prevent multi-firing glitch
LAST_TRADED_BARS: Dict[str, str] = {}

# ══════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════
DEFAULT_CONFIG = {
    "live_trading_enabled": False,
    "trading_pairs_1h": [
        "SOL/USDT","BTC/USDT","ETH/USDT"
    ],
    "trading_pairs_4h": [
        "SOL/USDT","BTC/USDT","ETH/USDT"
    ],
    "leverage_1h": 15,
    "leverage_4h": 10,
    "trade_size_pct": 50,
    "account_balance_usdc": 1000,
    "ema_length": 50,
    "rsi_length": 14,
    "volume_avg_length": 20,
    "volume_mult": 1.5,
    "candle_delta_length": 3,
    "candle_stability_mult": 0.8,
    "disable_repeating": True,
    "breakeven_r": 1.0,
    "trail_start_r": 2.0,
    "trail_atr_mult": 1.0,
    "trail_notify_min_move_pct": 0.5,
    "slippage_bps": 50,
    "hf_dataset_repo": (
        "sol-matrix-bot/cryptoai-state-data"
    ),
    "performance_metrics": {
        "total_signals": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "win_rate_pct": 0.0,
        "total_pnl_pct": 0.0,
        "total_fees_paid": 0.0
    }
}

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        if "performance_metrics" not in cfg:
            cfg["performance_metrics"] = \
                DEFAULT_CONFIG["performance_metrics"].copy()
        return cfg
    except Exception as e:
        log.error(f"Config error: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict):
    try:
        with portalocker.Lock(CONFIG_PATH, timeout=5):
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
    except Exception as e:
        log.error(f"Config save error: {e}")

def load_active_trades() -> Dict:
    try:
        with portalocker.Lock(
            ACTIVE_TRADE_PATH, timeout=5
        ):
            with open(ACTIVE_TRADE_PATH, "r") as f:
                return json.load(f)
    except:
        return {}

def save_active_trades(data: Dict):
    try:
        with portalocker.Lock(
            ACTIVE_TRADE_PATH, timeout=5
        ):
            with open(ACTIVE_TRADE_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        log.error(f"Trade save error: {e}")

# ══════════════════════════════════════════════
#  HUGGING FACE
# ══════════════════════════════════════════════
def save_to_hf(config: Dict):
    if not HF_TOKEN:
        return
    try:
        api     = HfApi(token=HF_TOKEN)
        payload = json.dumps(
            config.get("performance_metrics", {}),
            indent=2
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            f.write(payload)
            tmp = f.name
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="performance_metrics.json",
            repo_id=config.get(
                "hf_dataset_repo",
                "sol-matrix-bot/cryptoai-state-data"
            ),
            repo_type="dataset"
        )
        os.unlink(tmp)
        log.info("✅ Metrics synced to HF")
    except Exception as e:
        log.warning(f"HF sync error: {e}")

# ══════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════
def _sync_message(
    token: str, chat_id: str, text: str
) -> Tuple[int, dict]:
    url  = (
        f"https://api.telegram.org/"
        f"bot{token.strip()}/sendMessage"
    )
    resp = requests.post(
        url,
        json={
            "chat_id":    chat_id.strip(),
            "text":       text,
            "parse_mode": "HTML"
        },
        timeout=15
    )
    return resp.status_code, resp.json()

def _sync_photo(
    token: str, chat_id: str,
    path: str, caption: str
) -> Tuple[int, dict]:
    url = (
        f"https://api.telegram.org/"
        f"bot{token.strip()}/sendPhoto"
    )
    with open(path, 'rb') as p:
        resp = requests.post(
            url,
            data={
                "chat_id":    chat_id.strip(),
                "caption":    caption,
                "parse_mode": "HTML"
            },
            files={"photo": p},
            timeout=25
        )
    return resp.status_code, resp.json()

async def send_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("⚠️ Telegram credentials missing!")
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(
                _sync_message,
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                text
            )
            if status == 200:
                log.info("✅ Telegram sent!")
                return
            log.warning(
                f"⚠️ Telegram {status}: "
                f"{data.get('description','')}"
            )
        except Exception as e:
            log.warning(
                f"⚠️ Telegram attempt {attempt+1}: {e}"
            )
            await asyncio.sleep(5)

async def send_photo(path: str, caption: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if not os.path.exists(path):
        await send_message(caption)
        return
    for attempt in range(3):
        try:
            status, data = await asyncio.to_thread(
                _sync_photo,
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                path,
                caption
            )
            if status == 200:
                log.info("✅ Chart sent!")
                return
            log.warning(
                f"⚠️ Photo {status}: "
                f"{data.get('description','')}"
            )
        except Exception as e:
            log.warning(
                f"⚠️ Photo attempt {attempt+1}: {e}"
            )
            await asyncio.sleep(5)
    await send_message(caption)

# ══════════════════════════════════════════════
#  MATPLOTLIB CHART GENERATOR
# ══════════════════════════════════════════════
def generate_chart(df: pd.DataFrame, symbol: str, signal: str) -> str:
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        # FIXED ISSUE 1: Replaced invalid .suffix(30) with pandas standard .tail(30)
        plot_df = df.tail(30) if len(df) > 30 else df
        ax1.plot(plot_df['timestamp'], plot_df['close'], color='#58a6ff', label='Price', linewidth=2)
        ax1.plot(plot_df['timestamp'], plot_df['ema50'], color='#ff7b72', label='EMA 50', linestyle='--')
        
        c = '#56d364' if signal == 'BUY' else '#f85149'
        ax1.axhline(y=plot_df['close'].iloc[-1], color=c, linestyle=':', alpha=0.8, label=f'Signal Entry ({signal})')
        ax1.legend(loc='upper left', framealpha=0.2)
        ax1.set_title(f"{symbol} - Live Signal Geometry Trigger", color=FG, fontsize=12)

        ax2.bar(plot_df['timestamp'], plot_df['volume'], color='#30363d', alpha=0.7)
        if "volume_sma" in plot_df.columns:
            ax2.plot(plot_df['timestamp'], plot_df['volume_sma'], color='#d29922', label='Volume SMA')
        ax2.legend(loc='upper left', framealpha=0.2)

        plt.tight_layout()
        tmp_img = os.path.join(tempfile.gettempdir(), f"{symbol.replace('/', '_')}_signal.png")
        plt.savefig(tmp_img, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        return tmp_img
    except Exception as e:
        log.error(f"Error generating chart asset: {e}")
        return ""

# ══════════════════════════════════════════════
#  FEE CALCULATOR
# ══════════════════════════════════════════════
def calculate_fees(
    trade_size_usdc: float,
    sol_price_usdc: float
) -> Dict:
    open_fee  = trade_size_usdc * JUPITER_FEE_PCT
    close_fee = trade_size_usdc * JUPITER_FEE_PCT
    sol_fees  = SOL_NETWORK_FEE * 2 * sol_price_usdc
    total     = open_fee + close_fee + sol_fees
    return {
        "open_fee":   round(open_fee, 4),
        "close_fee":  round(close_fee, 4),
        "sol_fees":   round(sol_fees, 4),
        "total_fees": round(total, 4)
    }

def calculate_breakeven_price(
    entry: float,
    signal: str,
    fees: Dict,
    trade_size_usdc: float,
    leverage: int
) -> float:
    fee_pct    = fees["total_fees"] / trade_size_usdc
    price_move = entry * fee_pct / leverage
    if signal == "BUY":
        return round(entry + price_move, 6)
    else:
        return round(entry - price_move, 6)

# ══════════════════════════════════════════════
#  MARKET DATA
# ══════════════════════════════════════════════
COINGECKO_IDS = {
    "SOLUSDT": "solana",
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum"
}
KRAKEN_PAIRS = {
    "SOLUSDT": "SOLUSD",
    "BTCUSDT": "XBTUSD",
    "ETHUSDT": "ETHUSD"
}
BINANCE_WS_SYMBOLS = {
    "SOL/USDT": "solusdt",
    "BTC/USDT": "btcusdt",
    "ETH/USDT": "ethusdt"
}
KRAKEN_INTERVALS = {"1h": 60, "4h": 240}

async def _coingecko(
    clean: str, days: int = 14
) -> Optional[pd.DataFrame]:
    coin_id = COINGECKO_IDS.get(clean)
    if not coin_id:
        return None
    url = (
        f"https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/ohlc?vs_currency=usd&days={days}"
    )
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                url,
                timeout=aiohttp.ClientTimeout(total=12)
            ) as r:
                if r.status == 200:
                    raw = await r.json()
                    df  = pd.DataFrame(
                        raw,
                        columns=[
                            "timestamp","open",
                            "high","low","close"
                        ]
                    )
                    df = df.astype(float)
                    # FIXED ISSUE 4: Set to 100.0 flat scalar fallback
                    df["volume"]    = 100.0
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"].astype(int),
                        unit="ms"
                    )
                    return df.sort_values(
                        "timestamp"
                    ).reset_index(drop=True)
                log.warning(
                    f"⚠️ CoinGecko {r.status}"
                )
    except Exception as e:
        log.warning(f"⚠️ CoinGecko error: {e}")
    return None

async def _kraken(
    clean: str, tf: str = "1h"
) -> Optional[pd.DataFrame]:
    pair     = KRAKEN_PAIRS.get(clean)
    interval = KRAKEN_INTERVALS.get(tf, 60)
    if not pair:
        return None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.kraken.com/0/public/OHLC",
                params={
                    "pair":     pair,
                    "interval": interval
                },
                timeout=aiohttp.ClientTimeout(total=12)
            ) as r:
                d = await r.json()
                if d.get("error"):
                    raise ValueError(str(d["error"]))
                key  = list(d["result"].keys())[0]
                rows = d["result"][key]
                df   = pd.DataFrame(rows, columns=[
                    "timestamp","open","high","low",
                    "close","vwap","volume","count"
                ])
                df = df[[
                    "timestamp","open","high",
                    "low","close","volume"
                ]].astype({
                    "open": float,"high": float,
                    "low": float,"close": float,
                    "volume": float
                })
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"].astype(int),
                    unit="s"
                )
                return df.sort_values(
                    "timestamp"
                ).reset_index(drop=True)
    except Exception as e:
        log.warning(f"⚠️ Kraken error: {e}")
    return None

async def fetch_candles(
    symbol: str, tf: str = "1h"
) -> Optional[pd.DataFrame]:
    clean = symbol.replace("/","").upper()
    days  = 14 if tf == "1h" else 60

    df = await _coingecko(clean, days)
    if df is not None and len(df) >= 30:
        log.info(
            f"✅ CoinGecko {symbol} {tf} ({len(df)} candles)"
        )
        return df

    log.warning(
        f"⚠️ CoinGecko failed → Kraken {symbol}"
    )
    df = await _kraken(clean, tf)
    if df is not None and len(df) >= 30:
        log.info(
            f"✅ Kraken {symbol} {tf} ({len(df)} candles)"
        )
        return df

    log.error(f"🚨 All data failed {symbol} {tf}")
    return None

async def fetch_sol_price() -> float:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT", timeout=5) as r:
                if r.status == 200:
                    return float((await r.json())["price"])
    except:
        pass
    return 140.0 

# ══════════════════════════════════════════════
#  MACRO ENGINE INSIGHTS
# ══════════════════════════════════════════════
async def fetch_sentiment() -> float:
    vader = SentimentIntensityAnalyzer()
    headlines = []
    feeds = ["https://cointelegraph.com/rss", "https://decrypt.co/feed"]
    async with aiohttp.ClientSession() as s:
        for url in feeds:
            try:
                async with s.get(url, timeout=5) as r:
                    html = await r.text()
                    found = re.findall(r'<title>(.*?)</title>', html)[2:10]
                    headlines.extend(found)
            except: 
                pass
    if not headlines: 
        return 0.0
    return round(sum([vader.polarity_scores(h)["compound"] for h in headlines]) / len(headlines), 4)

async def fetch_fear_greed() -> Dict:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://api.alternative.me/fng/", timeout=5) as r:
                if r.status == 200:
                    d = (await r.json())["data"][0]
                    return {"value": int(d["value"]), "classification": d["value_classification"]}
    except: 
        pass
    return {"value": 50, "classification": "Neutral"}

# ══════════════════════════════════════════════
#  WEBSOCKET PRICE MONITOR
# ══════════════════════════════════════════════
async def websocket_price_stream(
    symbol: str,
    price_queue: asyncio.Queue
):
    ws_symbol = BINANCE_WS_SYMBOLS.get(symbol, "")
    if not ws_symbol:
        log.error(
            f"❌ No WebSocket symbol for {symbol}"
        )
        return

    url = (
        f"wss://stream.binance.com:9443/ws/"
        f"{ws_symbol}@trade"
    )
    log.info(
        f"🔌 WebSocket connecting {symbol}..."
    )

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                log.info(
                    f"✅ WebSocket connected {symbol}"
                )
                async for msg in ws:
                    try:
                        data  = json.loads(msg)
                        price = float(data["p"])
                        await price_queue.put(price)
                    except:
                        pass
        except Exception as e:
            log.warning(
                f"⚠️ WebSocket {symbol} error: {e}"
                f" — reconnecting in 5s..."
            )
            await asyncio.sleep(5)

# ══════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════
def add_indicators(
    df: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    df  = df.copy()
    rl  = config.get("rsi_length", 14)
    el  = config.get("ema_length", 50)
    vl  = config.get("volume_avg_length", 20)
    
    df["rsi"]      = RSIIndicator(
                         df["close"], window=rl
                     ).rsi()
    df["ema50"]    = EMAIndicator(
                         df["close"], window=el
                     ).ema_indicator()
    df["atr"]      = AverageTrueRange(
                         df["high"], df["low"],
                         df["close"], window=14
                     ).average_true_range()
    df["volume_sma"] = df["volume"].rolling(window=vl).mean()
    macd           = MACD(df["close"])
    df["macd"]     = macd.macd()
    bb             = BollingerBands(df["close"])
    df["bb_width"] = bb.bollinger_wband()
    return df.dropna().reset_index(drop=True)

# ══════════════════════════════════════════════
#  GAINZALGO V2 SIGNAL (EXACT LOGIC)
# ══════════════════════════════════════════════
def gainzalgo_signal(
    df: pd.DataFrame, config: Dict
) -> Optional[str]:
    if len(df) < 25:
        return None

    cdl = config.get("candle_delta_length", 3)
    csm = config.get("candle_stability_mult", 0.8)
    v_mult = config.get("volume_mult", 1.5)

    c   = df.iloc[-1]
    p   = df.iloc[-2]
    atr = c["atr"]

    candle_body = abs(c["close"] - c["open"])
    is_stable   = (candle_body / atr) > csm

    # FIXED ISSUE 4 Strategy: Bypasses check if data lacks genuine volume properties (CoinGecko mitigation)
    if df["volume"].nunique() > 1:
        volume_confirmed = c["volume"] > (c["volume_sma"] * v_mult)
    else:
        volume_confirmed = True

    if len(df) < cdl + 1:
        return None

    momentum_up   = (
        c["close"] > df.iloc[-(cdl+1)]["close"]
    )
    momentum_down = (
        c["close"] < df.iloc[-(cdl+1)]["close"]
    )

    # BUY
    fast_bull   = (
        c["close"] > p["high"]
        and c["close"] > c["open"]
    )
    rsi_bull    = c["rsi"] > 50
    is_buy      = (
        fast_bull
        and is_stable
        and rsi_bull
        and momentum_up
        and volume_confirmed
    )

    # SELL
    fast_bear   = (
        c["close"] < p["low"]
        and c["close"] < c["open"]
    )
    rsi_bear    = c["rsi"] < 50
    is_sell     = (
        fast_bear
        and is_stable
        and rsi_bear
        and momentum_down
        and volume_confirmed
    )

    if is_buy:  return "BUY"
    if is_sell: return "SELL"
    return None

# ══════════════════════════════════════════════
#  SL CALCULATION
# ══════════════════════════════════════════════
def calc_sl(
    df: pd.DataFrame, signal: str
) -> float:
    c   = df.iloc[-1]
    p   = df.iloc[-2]
    atr = c["atr"]
    if signal == "BUY":
        return round(p["low"] - atr * 0.5, 6)
    else:
        return round(p["high"] + atr * 0.5, 6)

# ══════════════════════════════════════════════
#  AI CONFIDENCE SCORER
# ══════════════════════════════════════════════
async def get_confidence(
    df: pd.DataFrame,
    signal: str,
    fg: Dict,
    sentiment: float
) -> Tuple[float, Dict]:
    row    = df.iloc[-1]
    scores = {}
    total  = 0.0

    # RSI (25%)
    rsi = row.get("rsi", 50)
    if signal == "BUY":
        if rsi > 65:
            scores["RSI"] = ("Strong ✅", 25)
        elif rsi > 55:
            scores["RSI"] = ("Good 🟡", 15)
        else:
            scores["RSI"] = ("Weak ⚠️", 5)
    else:
        if rsi < 35:
            scores["RSI"] = ("Strong ✅", 25)
        elif rsi < 45:
            scores["RSI"] = ("Good 🟡", 15)
        else:
            scores["RSI"] = ("Weak ⚠️", 5)
    total += scores["RSI"][1]

    # EMA50 (20%)
    ema50 = row.get("ema50", 0)
    price = row["close"]
    dist  = abs(price - ema50) / ema50 * 100
    if signal == "BUY" and price > ema50:
        scores["EMA50"] = (
            "Strong ✅", 20
        ) if dist > 2 else ("Good 🟡", 12)
    elif signal == "SELL" and price < ema50:
        scores["EMA50"] = (
            "Strong ✅", 20
        ) if dist > 2 else ("Good 🟡", 12)
    else:
        scores["EMA50"] = ("Weak ⚠️", 0)
    total += scores["EMA50"][1]

    # Fear & Greed (20%)
    fgv = fg.get("value", 50)
    if signal == "BUY":
        if fgv < 25:
            scores["Fear/Greed"] = (
                "Extreme Fear ✅", 20
            )
        elif fgv < 40:
            scores["Fear/Greed"] = ("Fear 🟡", 12)
        else:
            scores["Fear/Greed"] = ("Neutral ⚠️", 5)
    else:
        if fgv > 75:
            scores["Fear/Greed"] = (
                "Extreme Greed ✅", 20
            )
        elif fgv > 60:
            scores["Fear/Greed"] = ("Greed 🟡", 12)
        else:
            scores["Fear/Greed"] = ("Neutral ⚠️", 5)
    total += scores["Fear/Greed"][1]

    # Sentiment (20%)
    if signal == "BUY":
        if sentiment > 0.3:
            scores["Sentiment"] = ("Bullish ✅", 20)
        elif sentiment > 0.1:
            scores["Sentiment"] = ("Mild 🟡", 12)
        else:
            scores["Sentiment"] = ("Weak ⚠️", 5)
    else:
        if sentiment < -0.3:
            scores["Sentiment"] = ("Bearish ✅", 20)
        elif sentiment < -0.1:
            scores["Sentiment"] = ("Mild 🟡", 12)
        else:
            scores["Sentiment"] = ("Weak ⚠️", 5)
    total += scores["Sentiment"][1]

    # Candle strength (15%)
    atr   = row.get("atr", 0)
    body  = abs(row["close"] - row["open"])
    ratio = body / atr if atr > 0 else 0
    if ratio > 1.5:
        scores["Candle"] = ("Very Strong ✅", 15)
    elif ratio > 0.8:
        scores["Candle"] = ("Strong 🟡", 10)
    else:
        # FIXED TRUNCATION: Restored complete functionality block
        scores["Candle"] = ("Weak ⚠️", 5)
    total += scores["Candle"][1]

    return round(total, 1), scores

# ══════════════════════════════════════════════
#  REAL-TIME TICK MONITOR & STATE MACHINE
# ══════════════════════════════════════════════
async def stream_and_monitor_trade(symbol: str, tf: str, active_trade: Dict):
    ws_symbol = BINANCE_WS_SYMBOLS.get(symbol, "")
    if not ws_symbol: 
        return
    url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@trade"

    entry = active_trade["entry_price"]
    signal = active_trade["signal_type"]
    sl = active_trade["stop_loss"]
    be_target = active_trade["breakeven_target"]
    trail_trigger = active_trade["trail_trigger"]
    atr_val = active_trade["atr"]
    
    config = load_config()
    leverage = config.get(f"leverage_{tf}", 10)
    size_usdc = active_trade["trade_size_usdc"]

    log.info(f"🛰️ Spawning WebSocket Execution Listener Thread for {symbol}")
    
    # FIXED ISSUE 3: Wrapped with reconnection logic to prevent structural monitoring loss
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    current_price = float(data["p"])
                    
                    price_change_pct = ((current_price - entry) / entry) * 100 if signal == "BUY" else ((entry - current_price) / entry) * 100
                    gross_pnl_usdc = size_usdc * (price_change_pct / 100) * leverage

                    # ── 1. STOP LOSS TERMINATION CHECK ──
                    if (signal == "BUY" and current_price <= sl) or (signal == "SELL" and current_price >= sl):
                        net_pnl_usdc = gross_pnl_usdc - active_trade["fees"]["total_fees"]
                        msg = f"❌ <b>STOP LOSS HIT ({symbol} {tf})</b>\n\nExit Price: {current_price}\nPnL: ${round(net_pnl_usdc, 2)} ({round(price_change_pct * leverage, 2)}%)"
                        await send_message(msg)
                        
                        trades = load_active_trades()
                        trades.pop(f"{symbol}_{tf}", None)
                        save_active_trades(trades)
                        
                        config["performance_metrics"]["losses"] += 1
                        config["performance_metrics"]["total_signals"] += 1
                        config["performance_metrics"]["total_fees_paid"] += active_trade["fees"]["total_fees"]
                        save_config(config)
                        save_to_hf(config)
                        return # Exit tracking sequence gracefully

                    # ── 2. BREAKEVEN PROTECTOR CIRCUIT ──
                    if not active_trade.get("breakeven_hit", False):
                        if (signal == "BUY" and current_price >= be_target) or (signal == "SELL" and current_price <= be_target):
                            sl = active_trade["breakeven_price"]
                            active_trade["breakeven_hit"] = True
                            active_trade["stop_loss"] = sl
                            await send_message(f"🛡️ <b>BREAKEVEN TRIGGERED ({symbol})</b>\nStop loss insulated at parity line: {sl}")
                            trades = load_active_trades()
                            trades[f"{symbol}_{tf}"] = active_trade
                            save_active_trades(trades)

                    # ── 3. ATR TRAILING REVOLUTION SUB-ENGINE ──
                    if (signal == "BUY" and current_price >= trail_trigger) or (signal == "SELL" and current_price <= trail_trigger):
                        new_sl = round(current_price - atr_val * config.get("trail_atr_mult", 1.0), 6) if signal == "BUY" else round(current_price + atr_val * config.get("trail_atr_mult", 1.0), 6)
                        
                        if (signal == "BUY" and new_sl > sl) or (signal == "SELL" and new_sl < sl):
                            sl = new_sl
                            active_trade["stop_loss"] = sl
                            active_trade["trail_trigger"] = current_price + (atr_val * 0.5) if signal == "BUY" else current_price - (atr_val * 0.5)
                            await send_message(f"📈 <b>TRAILING SL UPDATE ({symbol})</b>\nNew Safeguard Target Floor: {sl}")
                            trades = load_active_trades()
                            trades[f"{symbol}_{tf}"] = active_trade
                            save_active_trades(trades)
        except Exception as e:
            log.warning(f"⚠️ Monitor stream error for {symbol}: {e}. Restoring connection loop in 5s...")
            await asyncio.sleep(5)

# ══════════════════════════════════════════════
#  EVALUATION LAYER
# ══════════════════════════════════════════════
async def process_pair(symbol: str, tf: str):
    global LAST_TRADED_BARS
    config = load_config()
    
    active_trades = load_active_trades()
    trade_key = f"{symbol}_{tf}"
    if trade_key in active_trades:
        return

    df = await fetch_candles(symbol, tf)
    if df is None or len(df) < 25: 
        return
    df = add_indicators(df, config)
    
    # ANTI-REPAINTING TRACKER: Identify uniquely forming candle instances
    live_bar_id = str(df.iloc[-1]["timestamp"])
    if LAST_TRADED_BARS.get(trade_key) == live_bar_id:
        return 

    signal = gainzalgo_signal(df, config)
    if not signal: 
        return

    # Lock processing parameters instantly to kill multi-firing glitch
    LAST_TRADED_BARS[trade_key] = live_bar_id

    entry = df.iloc[-1]["close"]
    atr_val = df.iloc[-1]["atr"]
    leverage = config.get(f"leverage_{tf}", 10)
    balance = config.get("account_balance_usdc", 1000.0)
    allocated_size = balance * (config.get("trade_size_pct", 50) / 100)
    
    sol_price = await fetch_sol_price()
    fees = calculate_fees(allocated_size, sol_price)
    be_price = calculate_breakeven_price(entry, signal, fees, allocated_size, leverage)
    stop_loss = calc_sl(df, signal)
    
    one_r_dist = abs(entry - stop_loss)
    be_target = entry + one_r_dist if signal == "BUY" else entry - one_r_dist
    trail_trigger = entry + (one_r_dist * config.get("trail_start_r", 2.0)) if signal == "BUY" else entry - (one_r_dist * config.get("trail_start_r", 2.0))

    fng = await fetch_fear_greed()
    sentiment = await fetch_sentiment()
    confidence, Breakdown = await get_confidence(df, signal, fng, sentiment)

    label = "🔥 VERY HIGH" if confidence >= 90 else ("✅ HIGH" if confidence >= 75 else ("🟡 MEDIUM" if confidence >= 60 else "⚠️ LOW"))
    alert_text = (
        f"🚀 <b>GAINZALGO GEOMETRY SIGNAL FIRED ({symbol} {tf})</b>\n\n"
        f"Action: <b>{signal}</b>\nEntry Floor: {entry}\nStop Loss: {stop_loss}\n"
        f"Leverage Profile: {leverage}x\nAllocated Size: ${round(allocated_size, 2)}\n\n"
        f"AI Confidence Score: <b>{confidence}% ({label})</b>\n"
        f"Fear & Greed Index: {fng['value']} ({fng['classification']})\n"
        f"Vader RSS Sentiment Compound: {sentiment}\n\n"
        f"Entry Friction Estimates:\n"
        f"• Fees: ${fees['total_fees']} USDC\n"
        f"• Calibrated Breakeven Target: {be_price}"
    )

    chart_path = generate_chart(df, symbol, signal)
    await send_photo(chart_path, alert_text)
    if chart_path and os.path.exists(chart_path): 
        os.unlink(chart_path)

    trade_payload = {
        "symbol": symbol, "timeframe": tf, "signal_type": signal, "entry_price": entry,
        "stop_loss": stop_loss, "breakeven_target": be_target, "breakeven_price": be_price,
        "trail_trigger": trail_trigger, "atr": atr_val, "trade_size_usdc": allocated_size,
        "fees": fees, "breakeven_hit": False, "timestamp": str(datetime.utcnow())
    }
    
    active_trades[trade_key] = trade_payload
    save_active_trades(active_trades)
    
    asyncio.create_task(stream_and_monitor_trade(symbol, tf, trade_payload))

# ══════════════════════════════════════════════
#  RUNTIME ORCHESTRATION PIPELINE
# ══════════════════════════════════════════════
async def main_loop():
    log.info("🤖 Starting GainzAlgo V2 Core Tracker Deployment Routine...")
    
    active_trades = load_active_trades()
    for key, payload in active_trades.items():
        asyncio.create_task(stream_and_monitor_trade(payload["symbol"], payload["timeframe"], payload))

    while True:
        config = load_config()
        tasks = []
        for pair in config.get("trading_pairs_1h", []):
            tasks.append(process_pair(pair, "1h"))
        for pair in config.get("trading_pairs_4h", []):
            tasks.append(process_pair(pair, "4h"))
        
        await asyncio.gather(*tasks)
        # FIXED ISSUE 2: Switched pacing to 15 seconds to prevent request stacking and API degradation
        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main_loop())
