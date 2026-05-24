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
                    df["volume"]    = 1.0
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
    if df is not None and len(df) >= 60:
        log.info(
            f"✅ CoinGecko {symbol} {tf} "
            f"({len(df)} candles)"
        )
        return df

    log.warning(
        f"⚠️ CoinGecko failed → Kraken {symbol}"
    )
    df = await _kraken(clean, tf)
    if df is not None and len(df) >= 60:
        log.info(
            f"✅ Kraken {symbol} {tf} "
            f"({len(df)} candles)"
        )
        return df

    log.error(f"🚨 All data failed {symbol} {tf}")
    return None

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
    if len(df) < 5:
        return None

    cdl = config.get("candle_delta_length", 3)
    csm = config.get("candle_stability_mult", 0.8)

    c   = df.iloc[-1]
    p   = df.iloc[-2]
    atr = c["atr"]

    candle_body = abs(c["close"] - c["open"])
    is_stable   = (candle_body / atr) > csm

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
        scores["Candle"] = ("Weak ⚠️", 3)
    total += scores["Candle"][1]

    return round(total, 1), scores

def confidence_label(pct: float) -> str:
    if pct >= 90: return "🔥 VERY HIGH"
    if pct >= 75: return "✅ HIGH"
    if pct >= 60: return "🟡 MEDIUM"
    return "⚠️ LOW"

# ══════════════════════════════════════════════
#  NEWS SENTIMENT
# ══════════════════════════════════════════════
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://coindesk.com/arc/outboundfeeds/rss/"
]

async def fetch_sentiment() -> float:
    vader     = SentimentIntensityAnalyzer()
    headlines = []
    async with aiohttp.ClientSession() as s:
        for url in RSS_FEEDS:
            try:
                async with s.get(
                    url,
                    timeout=aiohttp.ClientTimeout(
                        total=8
                    )
                ) as r:
                    html  = await r.text()
                    found = re.findall(
                        r'<title>(.*?)</title>',
                        html
                    )[2:12]
                    headlines.extend(found)
            except:
                pass
    if not headlines:
        return 0.0
    scores = [
        vader.polarity_scores(h)["compound"]
        for h in headlines[:15]
    ]
    return round(sum(scores) / len(scores), 4)

# ══════════════════════════════════════════════
#  FEAR & GREED
# ══════════════════════════════════════════════
async def fetch_fear_greed() -> Dict:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://api.alternative.me/fng/",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as r:
                d   = await r.json()
                val = int(d["data"][0]["value"])
                cls = d["data"][0][
                    "value_classification"
                ]
                return {"value": val, "label": cls}
    except:
        return {"value": 50, "label": "Neutral"}

# ══════════════════════════════════════════════
#  CHART ENGINE
# ══════════════════════════════════════════════
def generate_chart(
    df:        pd.DataFrame,
    symbol:    str,
    tf:        str,
    signal:    str,
    entry:     float,
    sl:        float,
    be_price:  float,
    conf:      float
) -> str:
    chart = df.tail(80).copy().set_index("timestamp")
    chart.index = pd.DatetimeIndex(chart.index)
    path  = (
        f"/tmp/{symbol.replace('/','')}{tf}_chart.png"
    )
    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor=BG
        )

        # Price
        ax1.plot(
            chart.index, chart["close"],
            color="#58a6ff", linewidth=1.5,
            label="Price", zorder=3
        )
        ax1.axhline(
            entry, color="#f0e68c",
            linestyle="--", linewidth=1.8,
            label=f"Entry ${entry:,.4f}"
        )
        ax1.axhline(
            sl, color="#f85149",
            linestyle="--", linewidth=1.8,
            label=f"SL ${sl:,.4f}"
        )
        ax1.axhline(
            be_price, color="#8b949e",
            linestyle=":", linewidth=1.2,
            label=f"Breakeven ${be_price:,.4f}"
        )

        if "ema50" in chart.columns:
            ax1.plot(
                chart.index, chart["ema50"],
                color="#8b949e", linewidth=1,
                linestyle=":", label="EMA50",
                alpha=0.7
            )

        fill_c = (
            "#3fb950"
            if signal == "BUY"
            else "#f85149"
        )
        ax1.fill_between(
            chart.index, sl, entry,
            color=fill_c, alpha=0.08
        )
        ax1.set_facecolor(BG)
        ax1.tick_params(colors=FG)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")
        ax1.set_title(
            f"{'🚀' if signal == 'BUY' else '📉'} "
            f"{symbol} {tf.upper()} | "
            f"{signal} | "
            f"AI: {conf:.0f}% | "
            f"{datetime.now().strftime('%H:%M %d/%m/%Y')}",
            color=FG, fontsize=11, pad=10
        )
        ax1.legend(
            loc="upper left",
            facecolor="#161b22",
            labelcolor=FG, fontsize=8
        )
        ax1.grid(
            True, color="#21262d", linewidth=0.6
        )

        # RSI
        rsi_vals = (
            chart["rsi"]
            if "rsi" in chart.columns
            else pd.Series([50]*len(chart))
        )
        ax2.plot(
            chart.index, rsi_vals,
            color="#e3b341", linewidth=1.3
        )
        ax2.axhline(
            70, color="#f85149",
            linestyle="--", alpha=0.6
        )
        ax2.axhline(
            50, color="#8b949e",
            linestyle=":", alpha=0.5
        )
        ax2.axhline(
            30, color="#3fb950",
            linestyle="--", alpha=0.6
        )
        ax2.fill_between(
            chart.index, rsi_vals, 70,
            where=(rsi_vals >= 70),
            color="#f85149", alpha=0.2
        )
        ax2.fill_between(
            chart.index, rsi_vals, 30,
            where=(rsi_vals <= 30),
            color="#3fb950", alpha=0.2
        )
        ax2.set_facecolor(BG)
        ax2.tick_params(colors=FG)
        ax2.set_ylabel("RSI", color=FG, fontsize=9)
        ax2.set_ylim(0, 100)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")
        ax2.grid(
            True, color="#21262d", linewidth=0.6
        )

        plt.tight_layout(h_pad=0.5)
        plt.savefig(
            path, dpi=110,
            bbox_inches="tight",
            facecolor=BG
        )
        plt.close(fig)
        log.info(f"✅ Chart saved → {path}")
    except Exception as e:
        log.error(f"❌ Chart error: {e}")
        plt.close("all")
    return path

# ══════════════════════════════════════════════
#  JUPITER DEX EXECUTION
# ══════════════════════════════════════════════
async def execute_trade(
    symbol: str, signal: str,
    amount_usdc: float, config: Dict
) -> Optional[dict]:
    if not config.get("live_trading_enabled", False):
        log.info("📊 Signal only — no live trade")
        return None
    in_mint  = (
        USDC_MINT if signal == "BUY" else SOL_MINT
    )
    out_mint = (
        SOL_MINT  if signal == "BUY" else USDC_MINT
    )
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "https://quote-api.jup.ag/v6/quote",
                    params={
                        "inputMint":   in_mint,
                        "outputMint":  out_mint,
                        "amount": int(
                            amount_usdc * 1_000_000
                        ),
                        "slippageBps": config.get(
                            "slippage_bps", 50
                        )
                    },
                    timeout=aiohttp.ClientTimeout(
                        total=10
                    )
                ) as r:
                    quote = await r.json()
                    log.info(
                        f"✅ Jupiter quote {symbol}"
                    )
                    return quote
        except Exception as e:
            log.warning(
                f"⚠️ Jupiter attempt {attempt+1}: {e}"
            )
            await asyncio.sleep(5)
    return None

# ══════════════════════════════════════════════
#  TRADE MONITOR (WEBSOCKET)
# ══════════════════════════════════════════════
async def monitor_trade(
    trade_key: str,
    trade:     Dict,
    config:    Dict
):
    symbol   = trade["symbol"]
    tf       = trade["tf"]
    signal   = trade["signal"]
    entry    = trade["entry"]
    sl       = trade["sl"]
    atr      = trade["atr"]
    leverage = trade["leverage"]
    fees     = trade["fees"]
    be_price = trade["breakeven_price"]
    trade_sz = trade["trade_size_usdc"]

    trail_sl    = trade.get("trail_sl", sl)
    be_done     = trade.get("breakeven_done", False)
    sl_dist     = abs(entry - sl)
    last_trail  = trail_sl
    notify_min  = config.get(
        "trail_notify_min_move_pct", 0.5
    ) / 100

    log.info(
        f"👁️ Monitoring {symbol} {tf} "
        f"{signal} @ ${entry:.4f} "
        f"[WebSocket]"
    )

    # Start WebSocket price stream
    price_queue: asyncio.Queue = asyncio.Queue()
    ws_task = asyncio.create_task(
        websocket_price_stream(symbol, price_queue)
    )

    try:
        while True:
            try:
                # Get real-time price from WebSocket
                price = await asyncio.wait_for(
                    price_queue.get(),
                    timeout=30
                )
            except asyncio.TimeoutError:
                log.warning(
                    f"⚠️ No price for 30s {symbol}"
                )
                continue

            # Calculate R multiple
            if signal == "BUY":
                pnl_r = (price - entry) / sl_dist
            else:
                pnl_r = (entry - price) / sl_dist

            # Raw PnL %
            raw_pnl_pct = pnl_r * (
                1 / leverage
            ) * 100 * leverage

            # True PnL after fees
            true_pnl_usdc = (
                (raw_pnl_pct / 100) * trade_sz
                - fees["total_fees"]
            )
            true_pnl_pct = (
                true_pnl_usdc / trade_sz * 100
            )

            # ── Breakeven at 1R ──
            be_r = config.get("breakeven_r", 1.0)
            if not be_done and pnl_r >= be_r:
                trail_sl = entry
                be_done  = True

                # Save state
                trades = load_active_trades()
                if trade_key in trades:
                    trades[trade_key][
                        "trail_sl"
                    ] = trail_sl
                    trades[trade_key][
                        "breakeven_done"
                    ] = True
                    save_active_trades(trades)

                await send_message(
                    f"🛡️ <b>BREAKEVEN HIT — "
                    f"{symbol} {tf}</b>\n\n"
                    f"✅ SL moved to entry!\n"
                    f"🔒 Trade now RISK FREE!\n\n"
                    f"📍 Entry:   "
                    f"<code>${entry:,.4f}</code>\n"
                    f"🛑 New SL:  "
                    f"<code>${trail_sl:,.4f}</code>\n"
                    f"💰 Price:   "
                    f"<code>${price:,.4f}</code>\n\n"
                    f"📈 Raw P&L:  "
                    f"<b>{raw_pnl_pct:+.2f}%</b>\n"
                    f"💸 Fees:    "
                    f"-${fees['total_fees']:.4f}\n"
                    f"✅ True P&L: "
                    f"<b>{true_pnl_pct:+.2f}%</b>\n\n"
                    f"⏰ "
                    f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                )
                log.info(
                    f"🛡️ Breakeven set {symbol}"
                )

            # ── Trail SL at 2R ──
            trail_r = config.get("trail_start_r", 2.0)
            if pnl_r >= trail_r:
                mult = config.get(
                    "trail_atr_mult", 1.0
                )
                if signal == "BUY":
                    new_trail = price - (atr * mult)
                    moved_up  = new_trail > trail_sl
                    if moved_up:
                        trail_sl = new_trail
                else:
                    new_trail = price + (atr * mult)
                    moved_dn  = new_trail < trail_sl
                    if moved_dn:
                        trail_sl = new_trail

                # Notify if moved enough
                if trail_sl != last_trail:
                    move_pct = abs(
                        trail_sl - last_trail
                    ) / last_trail

                    if move_pct >= notify_min:
                        # Save state
                        trades = load_active_trades()
                        if trade_key in trades:
                            trades[trade_key][
                                "trail_sl"
                            ] = trail_sl
                            save_active_trades(trades)

                        locked_pct = (
                            abs(trail_sl - entry)
                            / entry * 100 * leverage
                        )

                        await send_message(
                            f"📈 <b>TRAIL SL MOVED — "
                            f"{symbol} {tf}</b>\n\n"
                            f"🔼 SL Updated!\n\n"
                            f"🛑 Old SL: "
                            f"<code>${last_trail:,.4f}</code>\n"
                            f"🛑 New SL: "
                            f"<code>${trail_sl:,.4f}</code>\n"
                            f"💰 Price:  "
                            f"<code>${price:,.4f}</code>\n\n"
                            f"🔒 Locked: "
                            f"<b>{locked_pct:+.2f}%</b>\n"
                            f"📊 P&L R:  "
                            f"<b>{pnl_r:.2f}R</b>\n\n"
                            f"⏰ "
                            f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                        )
                        last_trail = trail_sl
                        log.info(
                            f"📈 Trail SL → "
                            f"${trail_sl:.4f} {symbol}"
                        )

            # ── Check SL Hit ──
            sl_hit = (
                (
                    signal == "BUY"
                    and price <= trail_sl
                )
                or
                (
                    signal == "SELL"
                    and price >= trail_sl
                )
            )

            if sl_hit:
                ws_task.cancel()

                # Final PnL
                if signal == "BUY":
                    raw_pnl = (
                        (trail_sl - entry)
                        / entry * 100 * leverage
                    )
                else:
                    raw_pnl = (
                        (entry - trail_sl)
                        / entry * 100 * leverage
                    )

                final_usdc = (
                    (raw_pnl / 100) * trade_sz
                    - fees["total_fees"]
                )
                final_pct  = (
                    final_usdc / trade_sz * 100
                )

                if final_pct > 0.5:
                    result = "WIN ✅"
                elif final_pct < -0.5:
                    result = "LOSS ❌"
                else:
                    result = "BREAKEVEN 🟡"

                # Update metrics
                cfg = load_config()
                pm  = cfg.setdefault(
                    "performance_metrics",
                    DEFAULT_CONFIG[
                        "performance_metrics"
                    ].copy()
                )
                pm["total_signals"] = \
                    pm.get("total_signals", 0) + 1

                if final_pct > 0.5:
                    pm["wins"] = \
                        pm.get("wins", 0) + 1
                elif final_pct < -0.5:
                    pm["losses"] = \
                        pm.get("losses", 0) + 1
                else:
                    pm["breakeven"] = \
                        pm.get("breakeven", 0) + 1

                total = pm["total_signals"]
                pm["win_rate_pct"] = round(
                    pm["wins"] / total * 100
                    if total > 0 else 0, 2
                )
                pm["total_pnl_pct"] = round(
                    pm.get("total_pnl_pct", 0)
                    + final_pct, 2
                )
                pm["total_fees_paid"] = round(
                    pm.get("total_fees_paid", 0)
                    + fees["total_fees"], 4
                )
                save_config(cfg)
                save_to_hf(cfg)

                # Remove trade
                trades = load_active_trades()
                trades.pop(trade_key, None)
                save_active_trades(trades)

                emoji = (
                    "✅" if final_pct > 0.5
                    else "🟡" if abs(final_pct) < 0.5
                    else "❌"
                )

                await send_message(
                    f"{emoji} <b>TRADE CLOSED — "
                    f"{symbol} {tf}</b>\n\n"
                    f"🏆 Result: <b>{result}</b>\n\n"
                    f"📍 Entry:     "
                    f"<code>${entry:,.4f}</code>\n"
                    f"📤 Exit:      "
                    f"<code>${trail_sl:,.4f}</code>\n"
                    f"⚡ Leverage:  <b>{leverage}x</b>\n\n"
                    f"📊 Raw P&L:   "
                    f"<b>{raw_pnl:+.2f}%</b>\n"
                    f"💸 Open Fee:  "
                    f"-${fees['open_fee']:.4f}\n"
                    f"💸 Close Fee: "
                    f"-${fees['close_fee']:.4f}\n"
                    f"💸 Net Fees:  "
                    f"-${fees['total_fees']:.4f}\n"
                    f"─────────────────\n"
                    f"✅ True P&L:  "
                    f"<b>{final_pct:+.2f}%</b>\n"
                    f"💵 True P&L:  "
                    f"<b>${final_usdc:+.2f}</b>\n\n"
                    f"📊 Total Trades: {total}\n"
                    f"🎯 Win Rate: "
                    f"{pm['win_rate_pct']:.1f}%\n"
                    f"💹 Total P&L: "
                    f"{pm['total_pnl_pct']:+.2f}%\n"
                    f"💸 Total Fees: "
                    f"${pm['total_fees_paid']:.4f}\n\n"
                    f"⏰ "
                    f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
                )
                log.info(
                    f"🏁 Closed {symbol} "
                    f"{final_pct:+.2f}%"
                )
                return

    except asyncio.CancelledError:
        ws_task.cancel()
        log.info(
            f"🛑 Monitor cancelled {symbol}"
        )
    except Exception as e:
        ws_task.cancel()
        log.error(
            f"🚨 Monitor crash {symbol}: {e}"
        )

# ══════════════════════════════════════════════
#  SIGNAL SCANNER
# ══════════════════════════════════════════════
async def scan_timeframe(
    tf:           str,
    config:       Dict,
    last_signals: Dict,
    active_tasks: Dict
):
    pairs    = config.get(
        f"trading_pairs_{tf}",
        ["SOL/USDT","BTC/USDT","ETH/USDT"]
    )
    leverage = config.get(
        f"leverage_{tf}",
        15 if tf == "1h" else 10
    )
    balance  = config.get(
        "account_balance_usdc", 1000
    )
    size_pct = config.get("trade_size_pct", 50)
    trade_sz = balance * (size_pct / 100)

    disable_repeat = config.get(
        "disable_repeating", True
    )

    sentiment, fg = await asyncio.gather(
        fetch_sentiment(),
        fetch_fear_greed()
    )

    for pair in pairs:
        try:
            df = await fetch_candles(pair, tf)
            if df is None or len(df) < 10:
                continue

            df     = add_indicators(df, config)
            signal = gainzalgo_signal(df, config)

            if signal is None:
                log.info(f"⏸ {pair} {tf}: No signal")
                continue

            # Anti-repeat
            key      = f"{pair}_{tf}"
            last_sig = last_signals.get(key)
            if disable_repeat and last_sig == signal:
                log.info(
                    f"🔁 Repeat {signal} {pair} "
                    f"{tf} — skipped"
                )
                continue

            last_signals[key] = signal

            row   = df.iloc[-1]
            entry = row["close"]
            atr   = row["atr"]
            sl    = calc_sl(df, signal)

            # Fee calculation
            fees = calculate_fees(
                trade_sz,
                entry if "SOL" in pair else 1
            )
            be_price = calculate_breakeven_price(
                entry, signal,
                fees, trade_sz, leverage
            )

            # AI confidence
            conf, breakdown = await get_confidence(
                df, signal, fg, sentiment
            )
            conf_lbl = confidence_label(conf)

            log.info(
                f"🚀 {signal} {pair} {tf} | "
                f"Entry: ${entry:.4f} | "
                f"SL: ${sl:.4f} | "
                f"Conf: {conf:.0f}%"
            )

            # Chart
            chart = generate_chart(
                df, pair, tf, signal,
                entry, sl, be_price, conf
            )

            # Breakdown text
            bd_text = "\n".join([
                f"  {k}: {v[0]}"
                for k, v in breakdown.items()
            ])

            emoji = "🚀" if signal == "BUY" else "📉"
            mode  = (
                "🔴 AUTO TRADE"
                if config.get("live_trading_enabled")
                else "🟡 SIGNAL ONLY"
            )

            msg = (
                f"{emoji} <b>{signal} — "
                f"{pair} {tf.upper()}</b>\n\n"
                f"🤖 <b>AI Confidence: "
                f"{conf:.0f}% {conf_lbl}</b>\n\n"
                f"📍 Entry:      "
                f"<code>${entry:,.4f}</code>\n"
                f"🛑 Stop Loss:  "
                f"<code>${sl:,.4f}</code>\n"
                f"📊 Breakeven:  "
                f"<code>${be_price:,.4f}</code>\n"
                f"🎯 Trailing:   <b>Active</b>\n"
                f"⚡ Leverage:   <b>{leverage}x</b>\n\n"
                f"💰 Trade Size: "
                f"<b>${trade_sz:.2f} "
                f"({size_pct}%)</b>\n"
                f"💸 Open Fee:   "
                f"-${fees['open_fee']:.4f}\n"
                f"💸 Close Fee:  "
                f"~-${fees['close_fee']:.4f}\n"
                f"💸 Total Fees: "
                f"~-${fees['total_fees']:.4f}\n\n"
                f"🤖 <b>AI Breakdown:</b>\n"
                f"{bd_text}\n\n"
                f"📊 RSI:      <b>{row['rsi']:.1f}</b>\n"
                f"📈 EMA50:    "
                f"<code>${row['ema50']:,.2f}</code>\n"
                f"😨 Fear/Greed: "
                f"<b>{fg['value']} ({fg['label']})</b>\n"
                f"📰 Sentiment: "
                f"<b>{sentiment:+.3f}</b>\n\n"
                f"⚙️ Mode: {mode}\n"
                f"⏰ "
                f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
            )
            await send_photo(chart, msg)

            # Execute if live
            if config.get("live_trading_enabled"):
                await execute_trade(
                    pair, signal, trade_sz, config
                )

            # Save trade
            trade_key = (
                f"{pair}_{tf}_"
                f"{int(datetime.now().timestamp())}"
            )
            trade_data = {
                "symbol":           pair,
                "tf":               tf,
                "signal":           signal,
                "entry":            entry,
                "sl":               sl,
                "trail_sl":         sl,
                "atr":              atr,
                "leverage":         leverage,
                "trade_size_usdc":  trade_sz,
                "fees":             fees,
                "breakeven_price":  be_price,
                "breakeven_done":   False,
                "timestamp":        datetime.now().isoformat()
            }
            trades = load_active_trades()
            trades[trade_key] = trade_data
            save_active_trades(trades)

            # Start monitor
            task = asyncio.create_task(
                monitor_trade(
                    trade_key, trade_data, config
                )
            )
            active_tasks[trade_key] = task

        except Exception as e:
            log.error(
                f"🚨 Scan error {pair} {tf}: {e}"
            )

# ══════════════════════════════════════════════
#  RESTORE ON RESTART
# ══════════════════════════════════════════════
async def restore_monitors(
    config: Dict, active_tasks: Dict
):
    trades = load_active_trades()
    if not trades:
        return
    log.info(
        f"🔄 Restoring {len(trades)} trades..."
    )
    for key, trade in trades.items():
        task = asyncio.create_task(
            monitor_trade(key, trade, config)
        )
        active_tasks[key] = task
        log.info(
            f"✅ Restored: "
            f"{trade['symbol']} {trade['tf']}"
        )

# ══════════════════════════════════════════════
#  MAIN 24/7 LOOP
# ══════════════════════════════════════════════
async def main():
    config        = load_config()
    last_signals: Dict = {}
    active_tasks: Dict = {}

    log.info("🚀 GainzAlgo Bot Starting...")

    await restore_monitors(config, active_tasks)

    balance  = config.get(
        "account_balance_usdc", 1000
    )
    size_pct = config.get("trade_size_pct", 50)

    await send_message(
        "🤖 <b>GainzAlgo V2 Bot Online!</b>\n\n"
        f"📊 1H Pairs: "
        f"{', '.join(config.get('trading_pairs_1h', []))}\n"
        f"📊 4H Pairs: "
        f"{', '.join(config.get('trading_pairs_4h', []))}\n\n"
        f"⚡ 1H Leverage: "
        f"<b>{config.get('leverage_1h', 15)}x</b>\n"
        f"⚡ 4H Leverage: "
        f"<b>{config.get('leverage_4h', 10)}x</b>\n"
        f"💰 Account:    <b>${balance}</b>\n"
        f"📊 Trade Size: "
        f"<b>{size_pct}% = "
        f"${balance * size_pct / 100:.2f}</b>\n\n"
        f"🔌 Monitor: <b>WebSocket Real-Time</b>\n"
        f"💸 Fees: <b>Calculated per trade</b>\n\n"
        f"⚙️ Mode: "
        f"{'🔴 LIVE TRADING' if config.get('live_trading_enabled') else '🟡 SIGNAL ONLY'}\n\n"
        f"⏰ "
        f"{datetime.now().strftime('%H:%M %d/%m/%Y')}"
    )

    last_1h = 0
    last_4h = 0

    while True:
        try:
            config = load_config()
            now    = datetime.now()

            # 1H check
            curr_1h = now.replace(
                minute=0, second=0, microsecond=0
            ).timestamp()
            if curr_1h > last_1h:
                last_1h = curr_1h
                log.info("⏰ 1H candle — scanning...")
                await scan_timeframe(
                    "1h", config,
                    last_signals, active_tasks
                )

            # 4H check
            curr_4h = (
                now.replace(
                    minute=0, second=0,
                    microsecond=0
                ).timestamp() //
                (4 * 3600) * (4 * 3600)
            )
            if curr_4h > last_4h:
                last_4h = curr_4h
                log.info("⏰ 4H candle — scanning...")
                await scan_timeframe(
                    "4h", config,
                    last_signals, active_tasks
                )

            # Cleanup done tasks
            done = [
                k for k, t in active_tasks.items()
                if t.done()
            ]
            for k in done:
                del active_tasks[k]

            await asyncio.sleep(30)

        except Exception as e:
            log.error(f"🚨 Main error: {e}")
            await asyncio.sleep(60)

# ══════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 Bot stopped.")
