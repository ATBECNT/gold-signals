"""XAU/USD signal bot: candles -> signal rules -> news + sentiment checks -> Telegram."""
import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "XAU/USD"

# One setup per timeframe. The timeframe decides the trade type.
# sl_atr = stop distance in ATR multiples; tp_r = TP1/TP2/TP3 in multiples of the risk.
TIMEFRAMES = {
    "15min": {"label": "15m", "type": "SCALP",    "minutes": 15,  "sl_atr": 1.2, "tp_r": (1.0, 1.5, 2.0)},
    "1h":    {"label": "1h",  "type": "INTRADAY", "minutes": 60,  "sl_atr": 1.5, "tp_r": (1.0, 2.0, 3.0)},
    "4h":    {"label": "4h",  "type": "SWING",    "minutes": 240, "sl_atr": 2.0, "tp_r": (1.5, 3.0, 4.5)},
}

# ---- News / sentiment settings (tweak these) ----
BLACKOUT_BEFORE_MIN = 45     # no signals this long before a high-impact USD event
BLACKOUT_AFTER_MIN = 30      # ...and this long after it
WARN_AHEAD_HOURS = 12        # mention the next high-impact USD event if it is this close
CALENDAR_TTL_HOURS = 6       # how often to refresh the calendar
SENTIMENT_TTL_MIN = 60       # reuse a sentiment reading this long (free API: 25 calls/day)
SENTIMENT_BLOCK = 0.15       # skip a signal if gold sentiment is this strongly against it
SENTIMENT_MIN_ARTICLES = 3   # ignore sentiment if fewer gold articles than this
MAX_SIGNAL_AGE_MIN = 20      # ignore signals whose candle closed longer ago than this

# ---- New York open ORB settings ----
ORB_TZ = "America/New_York"
ORB_START = "09:30"          # opening range = the 15m candle that opens at 9:30 New York time
ORB_LAST_ENTRY = "11:30"     # a breakout candle must open before this (NY time)
ORB_MIN_ATR = 0.5            # skip if the range is smaller than this many ATRs (fakeout-prone)
ORB_MAX_ATR = 2.5            # skip if the range is wider than this (stop too far)
ORB_SL_BUFFER_ATR = 0.1      # SL sits this many ATRs beyond the opposite end of the range
ORB_TP_R = (1.0, 2.0, 3.0)
ORB_CFG = {"label": "NY Open ORB", "type": "INTRADAY", "minutes": 15}

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
GOLD_WORDS = re.compile(r"\b(gold|bullion|xau|precious metals?)\b", re.I)

STATE_FILE = Path("state.json")
LOG_FILE = Path("signals_log.csv")
CALENDAR_FILE = Path("calendar.json")
LOG_HEADER = ["logged_utc", "candle_utc", "timeframe", "type", "side", "entry",
              "sl", "tp1", "tp2", "tp3", "status", "sentiment"]


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ----------------------------- price data -----------------------------
def fetch(interval: str) -> pd.DataFrame:
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": SYMBOL, "interval": interval, "outputsize": 300,
            "order": "ASC", "timezone": "UTC",
            "apikey": os.environ["TWELVEDATA_API_KEY"],
        },
        timeout=30,
    )
    data = r.json()
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data error for {interval}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    # The last row is the candle still forming, so drop it and use closed candles only.
    return df.iloc[:-1].reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    prev_close = c.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def check_signal(df: pd.DataFrame, cfg: dict):
    """Trend-pullback rule: trade in the direction of the EMA trend when price
    closes back through the EMA20 after dipping on the other side of it."""
    if len(df) < 210:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    side = None
    if (last.ema20 > last.ema50 > last.ema200
            and prev.close < prev.ema20 and last.close > last.ema20
            and 40 <= last.rsi <= 65):
        side = "BUY"
    elif (last.ema20 < last.ema50 < last.ema200
            and prev.close > prev.ema20 and last.close < last.ema20
            and 35 <= last.rsi <= 60):
        side = "SELL"
    if side is None:
        return None

    entry = float(last.close)
    risk = float(last.atr) * cfg["sl_atr"]
    sign = 1 if side == "BUY" else -1
    return {
        "side": side,
        "entry": round(entry, 2),
        "sl": round(entry - sign * risk, 2),
        "tps": [round(entry + sign * risk * r, 2) for r in cfg["tp_r"]],
        "risk": round(risk, 2),
        "candle": last.datetime.strftime("%Y-%m-%d %H:%M"),
        "candle_dt": last.datetime,
    }


def check_orb(df: pd.DataFrame):
    """New York open breakout: first 15m candle that CLOSES outside the 9:30 candle's range."""
    ny = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(ORB_TZ)
    day = ny.iloc[-1].date()
    today = ny.dt.date == day
    d, t = df[today], ny[today]
    rng_rows = d[t.dt.strftime("%H:%M") == ORB_START]
    if rng_rows.empty:
        return None
    rng = rng_rows.iloc[0]
    last = df.iloc[-1]
    if last.datetime <= rng.datetime or ny.iloc[-1].strftime("%H:%M") >= ORB_LAST_ENTRY:
        return None
    after = d[d.datetime > rng.datetime]
    outside = after[(after.close > rng.high) | (after.close < rng.low)]
    # only the FIRST breakout counts, and only if it is the candle that just closed
    if outside.empty or outside.iloc[0].datetime != last.datetime:
        return None
    size = float(rng.high - rng.low)
    if not (ORB_MIN_ATR * rng.atr <= size <= ORB_MAX_ATR * rng.atr):
        return None

    side = "BUY" if last.close > rng.high else "SELL"
    sign = 1 if side == "BUY" else -1
    entry = float(last.close)
    sl = float(rng.low - ORB_SL_BUFFER_ATR * rng.atr) if side == "BUY" else float(rng.high + ORB_SL_BUFFER_ATR * rng.atr)
    risk = abs(entry - sl)
    return {
        "side": side,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tps": [round(entry + sign * risk * r, 2) for r in ORB_TP_R],
        "risk": round(risk, 2),
        "candle": last.datetime.strftime("%Y-%m-%d %H:%M"),
        "candle_dt": last.datetime,
        "extra": f"Opening range (9:30 NY): {rng.low:.2f} - {rng.high:.2f}",
    }


# ------------------------------- news ---------------------------------
def load_events():
    """High-impact USD events (UTC). Returns None if the calendar is unavailable."""
    cache = json.loads(CALENDAR_FILE.read_text()) if CALENDAR_FILE.exists() else {}
    now = now_utc()
    fetched = pd.Timestamp(cache["fetched_at"]) if cache.get("fetched_at") else None
    if fetched is None or now - fetched > pd.Timedelta(hours=CALENDAR_TTL_HOURS):
        try:
            r = requests.get(CALENDAR_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            events = []
            for e in r.json():
                if e.get("country") == "USD" and str(e.get("impact", "")).lower() == "high":
                    ts = pd.Timestamp(e["date"])
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                    events.append({"title": e["title"], "time": ts.isoformat()})
            cache = {"fetched_at": now.isoformat(), "events": events}
            CALENDAR_FILE.write_text(json.dumps(cache, indent=1))
        except Exception as e:
            print(f"calendar fetch failed: {e}")
            if not (fetched is not None and now - fetched < pd.Timedelta(hours=24)):
                return None  # no usable cache
    return cache.get("events", [])


def news_check(now: pd.Timestamp) -> dict:
    events = load_events()
    if events is None:
        return {"blocked": False, "note": "⚠ News calendar unavailable. Check the news yourself."}
    nearest = None
    for ev in events:
        mins = (pd.Timestamp(ev["time"]) - now).total_seconds() / 60
        if -BLACKOUT_AFTER_MIN <= mins <= BLACKOUT_BEFORE_MIN:
            return {"blocked": True, "note": f"{ev['title']} ({mins:+.0f} min)"}
        if 0 < mins <= WARN_AHEAD_HOURS * 60 and (nearest is None or mins < nearest[0]):
            nearest = (mins, ev["title"])
    if nearest:
        h, m = divmod(int(nearest[0]), 60)
        return {"blocked": False, "note": f"⚠ {nearest[1]} (USD, high impact) in {h}h {m:02d}m"}
    return {"blocked": False, "note": f"No high-impact USD news in the next {WARN_AHEAD_HOURS}h"}


# ----------------------------- sentiment ------------------------------
def sentiment_label(score: float) -> str:
    if score <= -0.35: return "Bearish"
    if score <= -0.15: return "Somewhat-Bearish"
    if score < 0.15:   return "Neutral"
    if score < 0.35:   return "Somewhat-Bullish"
    return "Bullish"


def gold_sentiment(state: dict, now: pd.Timestamp):
    """Average Alpha Vantage news sentiment of gold headlines in the last 24h."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return None
    cached = state.get("_sentiment")
    if cached and now - pd.Timestamp(cached["at"]) < pd.Timedelta(minutes=SENTIMENT_TTL_MIN):
        return cached
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets,economy_macro,economy_monetary",
                "time_from": (now - pd.Timedelta(hours=24)).strftime("%Y%m%dT%H%M"),
                "sort": "LATEST", "limit": 200, "apikey": key,
            },
            timeout=30,
        )
        data = r.json()
        if "feed" not in data:
            raise RuntimeError(str(data)[:200])
        scores = [float(a["overall_sentiment_score"]) for a in data["feed"]
                  if GOLD_WORDS.search(f"{a.get('title', '')} {a.get('summary', '')}")]
        result = {"at": now.isoformat(), "n": len(scores),
                  "score": round(sum(scores) / len(scores), 3) if scores else None}
    except Exception as e:
        print(f"sentiment fetch failed: {e}")
        return None
    state["_sentiment"] = result
    return result


# ----------------------------- telegram -------------------------------
def send_telegram(text: str) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
        data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text},
        timeout=30,
    )
    r.raise_for_status()


def format_message(sig: dict, cfg: dict, news_note: str, sent_line: str) -> str:
    icon = "🟢" if sig["side"] == "BUY" else "🔴"
    t1, t2, t3 = sig["tps"]
    return (
        f"{icon} {sig['side']} {SYMBOL} | {cfg['type']} ({cfg['label']})\n\n"
        f"Entry: ~{sig['entry']}\n"
        f"SL: {sig['sl']}  (risk {sig['risk']} pts)\n"
        f"TP1: {t1}\nTP2: {t2}\nTP3: {t3}\n"
        f"{sig['extra'] + chr(10) if sig.get('extra') else ''}\n"
        f"📰 {news_note}\n"
        f"🧭 {sent_line}\n\n"
        f"Candle closed: {(sig['candle_dt'] + pd.Timedelta(minutes=cfg['minutes'])).strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"Check the live price in MT5 before entering. Not financial advice."
    )


def log_row(row: list) -> None:
    with LOG_FILE.open("a", newline="") as f:
        csv.writer(f).writerow(row)


# -------------------------------- main --------------------------------
def handle(key: str, sig, cfg: dict, state: dict) -> None:
    """Run one candidate signal through age, news and sentiment checks, then send it."""
    if not sig or state.get(key) == sig["candle"]:
        print(f"{key}: no new signal")
        return
    state[key] = sig["candle"]  # each candle is evaluated only once
    now = now_utc()

    def log(status, sent="-"):
        log_row([now.strftime("%Y-%m-%d %H:%M"), sig["candle"], key, cfg["type"],
                 sig["side"], sig["entry"], sig["sl"], *sig["tps"], status, sent])

    closed_at = sig["candle_dt"] + pd.Timedelta(minutes=cfg["minutes"])
    if now.tz_localize(None) - closed_at > pd.Timedelta(minutes=MAX_SIGNAL_AGE_MIN):
        print(f"{key}: signal too old, ignored")
        return

    news = news_check(now)
    if news["blocked"]:
        print(f"{key}: skipped, news blackout: {news['note']}")
        log("SKIPPED_NEWS")
        return

    sent = gold_sentiment(state, now)
    if sent and sent["score"] is not None and sent["n"] >= SENTIMENT_MIN_ARTICLES:
        s = sent["score"]
        against = (sig["side"] == "BUY" and s <= -SENTIMENT_BLOCK) or \
                  (sig["side"] == "SELL" and s >= SENTIMENT_BLOCK)
        tag = f"{sentiment_label(s)} ({s:+.2f}, {sent['n']} gold articles/24h)"
        if against:
            print(f"{key}: skipped, sentiment against trade: {tag}")
            log("SKIPPED_SENTIMENT", s)
            return
        aligned = (sig["side"] == "BUY" and s >= SENTIMENT_BLOCK) or \
                  (sig["side"] == "SELL" and s <= -SENTIMENT_BLOCK)
        sent_line = f"Gold sentiment: {tag}, {'supports' if aligned else 'neutral for'} this trade"
        sent_log = s
    elif sent:
        sent_line, sent_log = "Gold sentiment: not enough recent gold news to judge", "-"
    else:
        sent_line, sent_log = "Gold sentiment: unavailable", "-"

    send_telegram(format_message(sig, cfg, news["note"], sent_line))
    log("SENT", sent_log)
    print(f"{key}: sent {sig['side']}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send_telegram("✅ Test message: your gold signal bot is connected.")
        return 0

    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    if not LOG_FILE.exists():
        LOG_FILE.write_text(",".join(LOG_HEADER) + "\n")

    errors = 0
    for interval, cfg in TIMEFRAMES.items():
        try:
            df = add_indicators(fetch(interval))
            handle(interval, check_signal(df, cfg), cfg, state)
            if interval == "15min":  # the NY-open ORB uses the same 15m candles
                handle("orb", check_orb(df), ORB_CFG, state)
        except Exception as e:  # keep going so one failure doesn't block other timeframes
            errors += 1
            print(f"{interval}: ERROR {e}")

    STATE_FILE.write_text(json.dumps(state, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
