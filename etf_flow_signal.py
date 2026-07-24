"""
etf_flow_signal.py
==================
Daily-resolution study of US spot Bitcoin ETF flows (Farside) vs BTC price.

What it does:
  1. Scrapes the full daily flow history from farside.co.uk (Jan 2024 -> today).
  2. Pulls daily BTC-USD closes (yfinance), or reads your own CSV if provided.
  2b. Splits flow into organic (all funds ex-GBTC) vs GBTC (legacy trust,
      mostly structural bleed) — the z-score/regime/quintile signal below
      is built on organic flow so GBTC's near-constant outflow doesn't
      drag down the trailing baseline and mislabel ordinary days.
  3. Joins the two on trading dates and runs:
       a. Cross-correlation of daily net flow vs daily log returns at lags
          -10..+10 (does flow lead price, or does price lead flow?)
       b. Conditional forward returns: what happens to BTC after big
          inflow / outflow days (quintile analysis, bucketed on a rolling
          flow z-score so a "big" day means the same thing in 2024 and
          2026 despite the ETF category's AUM growth). Reports moving-
          block-bootstrap 90% CIs alongside the means, since forward
          windows overlap and inflate naive significance.
       c. A regime gate: 20d slope of a z-scored cumulative flow series
          (scale-free, not raw $) + price vs 50d SMA
          -> LONG / RECOVERY / DISTRIBUTION / STAND_DOWN
       d. Divergence detector: continuous score = price's percentile rank
          in its 20d range minus flow's percentile rank in its 20d range
          (graded strength, not just a binary new-high/new-low flag).
  3b. Pulls BTCUSDT perpetual funding rate (Binance, free public endpoint)
      as a positioning/crowding read, and Deribit's DVOL index (BTC's
      30d implied-vol index) to compute the vol risk premium (IV - 20d
      realized vol) — the number that actually answers "is premium worth
      selling right now," which flow/price alone can't tell you. Network-
      optional: skip with --no-options-data, or a fetch failure just
      degrades gracefully (columns omitted, rest of the pipeline unaffected).
  4. Prints a signal summary, saves a chart (etf_flow_signal.png) and a
     daily CSV (etf_flow_daily.csv) you can feed into the dashboard.

Usage (Windows PowerShell):
  python etf_flow_signal.py
  python etf_flow_signal.py --btc-csv C:\\Users\\Adam\\BTC_Dashboard\\data\\btc_daily.csv

Requires: pandas numpy scipy matplotlib requests lxml yfinance
  pip install pandas numpy scipy matplotlib requests lxml yfinance
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime

import numpy as np
import pandas as pd

FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------

def parse_farside_table(html: str) -> pd.DataFrame:
    """Parse the Farside all-data page into a daily DataFrame.

    Handles the site's conventions:
      * negatives in parentheses, e.g. (95.1)
      * thousands commas, e.g. 1,045.0
      * '-' for holidays / not-yet-listed funds
      * a trailing 'Total' footer row that must be dropped
    Also splits flow into organic (all funds except GBTC) vs GBTC: GBTC is
    the legacy Grayscale trust that converted into an ETF on launch day and
    has bled almost continuously since (tax-loss-harvesting / arbitrage
    unwind, not fresh market conviction). Netting it out isolates flow that
    actually reflects new demand from that structural overhang.
    Returns columns: date (index, datetime), net_flow / cum_flow (Total,
    US$m), net_flow_organic / cum_flow_organic (ex-GBTC),
    net_flow_gbtc / cum_flow_gbtc (GBTC only).
    """
    tables = pd.read_html(io.StringIO(html))
    # The flow table is the one with a 'Date' and 'Total' column
    tbl = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Date" in cols and "Total" in cols:
            tbl = t
            break
    if tbl is None:
        raise ValueError("Could not locate flow table on Farside page")

    tbl.columns = [str(c).strip() for c in tbl.columns]

    # Drop footer 'Total' row and any non-date rows
    tbl["date"] = pd.to_datetime(tbl["Date"], format="%d %b %Y", errors="coerce")
    tbl = tbl.dropna(subset=["date"])

    def to_num(x) -> float:
        s = str(x).strip().replace(",", "")
        if s in ("-", "", "nan", "None"):
            return np.nan
        if s.startswith("(") and s.endswith(")"):
            return -float(s[1:-1])
        return float(s)

    fund_cols = [c for c in tbl.columns if c not in ("Date", "date", "Total")]
    for c in fund_cols + ["Total"]:
        tbl[c] = tbl[c].map(to_num)

    organic_cols = [c for c in fund_cols if c != "GBTC"]
    tbl["net_flow"] = tbl["Total"]
    tbl["net_flow_gbtc"] = tbl["GBTC"] if "GBTC" in tbl.columns else 0.0
    tbl["net_flow_organic"] = tbl[organic_cols].sum(axis=1, min_count=1)

    tbl = tbl[["date", "net_flow", "net_flow_organic", "net_flow_gbtc"]].set_index("date").sort_index()
    # Holiday rows come through as 0.0/NaN on Farside; keep them, they are
    # legitimate zero-flow trading-calendar rows and get dropped in the join
    # if BTC has no bar (BTC trades 24/7 so they will match anyway).
    tbl["cum_flow"] = tbl["net_flow"].fillna(0).cumsum()
    tbl["cum_flow_organic"] = tbl["net_flow_organic"].fillna(0).cumsum()
    tbl["cum_flow_gbtc"] = tbl["net_flow_gbtc"].fillna(0).cumsum()
    return tbl


def fetch_farside() -> pd.DataFrame:
    # cloudscraper (not plain requests) because farside.co.uk sits behind
    # Cloudflare bot protection that blocks known datacenter/CI IP ranges
    # (e.g. GitHub Actions runners) outright -- cloudscraper emulates a
    # real browser's TLS/JS fingerprint to get past that. Deliberately not
    # passing our own headers here: cloudscraper picks headers matched to
    # the TLS fingerprint it's emulating, and overriding them (e.g. the
    # custom UA) can make the two inconsistent and trip detection anyway.
    import cloudscraper

    scraper = cloudscraper.create_scraper()
    r = scraper.get(FARSIDE_URL, timeout=30)
    r.raise_for_status()
    return parse_farside_table(r.text)


def fetch_btc(btc_csv: str | None) -> pd.Series:
    """Daily BTC close. CSV needs columns date,close (any casing).
    Falls back to yfinance BTC-USD if no CSV is given."""
    if btc_csv:
        df = pd.read_csv(btc_csv)
        df.columns = [c.lower() for c in df.columns]
        date_col = next(c for c in df.columns if "date" in c or "time" in c)
        close_col = next(c for c in df.columns if "close" in c)
        s = pd.Series(
            df[close_col].values, index=pd.to_datetime(df[date_col]), name="btc_close"
        ).sort_index()
        return s
    import yfinance as yf

    px = yf.download("BTC-USD", start="2024-01-01", auto_adjust=True, progress=False)
    close = px["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance>=0.2.40 multiindex quirk
        close = close.iloc[:, 0]
    close.name = "btc_close"
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def fetch_funding_rate(start: str = "2024-01-01") -> pd.Series:
    """Daily BTCUSDT perpetual funding rate (Binance, summed from 8h prints).
    Positive = longs pay shorts: a crowded-long / bullish-carry-cost read
    that's orthogonal to spot flows and price (it's a derivatives
    positioning signal, not a spot-demand one)."""
    import requests

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "startTime": cur, "endTime": end_ms, "limit": 1000},
            headers=UA, timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.normalize()
    df["fundingRate"] = df["fundingRate"].astype(float)
    daily = df.groupby("date")["fundingRate"].sum()
    daily.name = "funding_daily"
    return daily


def fetch_dvol(start: str = "2024-01-01") -> pd.Series:
    """Daily BTC DVOL close (Deribit's 30d forward-looking implied-vol
    index — the BTC equivalent of the VIX). Unlike realized vol this is a
    market-priced expectation, so it leads rather than just describes."""
    import requests

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    r = requests.get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={"currency": "BTC", "start_timestamp": start_ms,
                "end_timestamp": end_ms, "resolution": "86400"},
        headers=UA, timeout=30,
    )
    r.raise_for_status()
    data = r.json()["result"]["data"]
    idx = pd.to_datetime([row[0] for row in data], unit="ms").normalize()
    close = [row[4] for row in data]
    s = pd.Series(close, index=idx, name="dvol", dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


# ----------------------------------------------------------------------------
# 2. Analytics
# ----------------------------------------------------------------------------

def build_frame(flows: pd.DataFrame, btc: pd.Series,
                 z_win: int = 90, z_min_periods: int = 40) -> pd.DataFrame:
    df = flows.join(btc, how="inner")
    df["log_ret"] = np.log(df["btc_close"]).diff()
    df["fwd_ret_1d"] = df["log_ret"].shift(-1)
    df["fwd_ret_5d"] = np.log(df["btc_close"].shift(-5) / df["btc_close"])
    df["fwd_ret_20d"] = np.log(df["btc_close"].shift(-20) / df["btc_close"])
    # Rolling z-score of daily ORGANIC (ex-GBTC) flow: typical flow $
    # magnitude has grown a lot since Jan 2024 as the ETF category's AUM
    # base grew, so a raw-$ threshold means something different in 2024 vs
    # now. Scoring against a trailing 90d mean/std keeps "big day"
    # comparable across the history. Using organic rather than Total flow
    # also keeps GBTC's near-constant legacy bleed from dragging the
    # trailing mean down and making ordinary days look artificially
    # "above average."
    df["flow_roll_mean"] = df["net_flow_organic"].rolling(z_win, min_periods=z_min_periods).mean()
    df["flow_roll_std"] = df["net_flow_organic"].rolling(z_win, min_periods=z_min_periods).std()
    df["flow_z"] = (df["net_flow_organic"] - df["flow_roll_mean"]) / df["flow_roll_std"]
    return df


def add_options_data(df: pd.DataFrame, funding: pd.Series | None, dvol: pd.Series | None,
                     z_win: int = 90, z_min_periods: int = 40) -> pd.DataFrame:
    """Join funding/DVOL onto the daily frame (left join: missing days on
    either source just leave those columns NaN rather than dropping rows)
    and derive the vol risk premium (IV - realized) plus a funding z-score
    using the same rolling-normalization approach as the flow z-score."""
    d = df.copy()
    d["realized_vol_20d"] = d["log_ret"].rolling(20).std() * np.sqrt(365) * 100
    if funding is not None:
        d = d.join(funding, how="left")
        d["funding_ann_pct"] = d["funding_daily"] * 365 * 100
        fr_mean = d["funding_daily"].rolling(z_win, min_periods=z_min_periods).mean()
        fr_std = d["funding_daily"].rolling(z_win, min_periods=z_min_periods).std()
        d["funding_z"] = (d["funding_daily"] - fr_mean) / fr_std
    if dvol is not None:
        d = d.join(dvol, how="left")
        d["iv_rv_spread"] = d["dvol"] - d["realized_vol_20d"]
    return d


def funding_quintile_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket days into funding-z quintiles; report forward returns with CIs.
    Tests funding as a crowding/contrarian signal, independent of flows."""
    if "funding_z" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["funding_z", "fwd_ret_1d"]).copy()
    d["q"] = pd.qcut(d["funding_z"], 5,
                     labels=["Q1 crowded short", "Q2", "Q3", "Q4", "Q5 crowded long"])
    rows = []
    for q, grp in d.groupby("q", observed=True):
        lo5, hi5 = block_bootstrap_mean_ci(grp["fwd_ret_5d"], block=5)
        lo20, hi20 = block_bootstrap_mean_ci(grp["fwd_ret_20d"], block=20)
        rows.append({
            "q": q,
            "n": len(grp),
            "avg_funding_ann_%": grp["funding_ann_pct"].mean(),
            "fwd_1d_bps": grp["fwd_ret_1d"].mean() * 1e4,
            "fwd_5d_bps": grp["fwd_ret_5d"].mean() * 1e4,
            "fwd_5d_ci90": f"[{lo5:+.0f}, {hi5:+.0f}]",
            "fwd_20d_bps": grp["fwd_ret_20d"].mean() * 1e4,
            "fwd_20d_ci90": f"[{lo20:+.0f}, {hi20:+.0f}]",
        })
    out = pd.DataFrame(rows).set_index("q")
    out.index.name = "q"
    return out.round(2)


def cross_correlation(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """corr(net_flow[t], log_ret[t+lag]).

    lag > 0  -> flow today vs FUTURE returns (flow leading price)
    lag < 0  -> flow today vs PAST returns  (price leading flow)
    """
    f = df["net_flow"]
    r = df["log_ret"]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        c = f.corr(r.shift(-lag))
        n = min(f.notna().sum(), r.shift(-lag).notna().sum())
        # ~2 std err significance band for a correlation of iid series
        rows.append({"lag": lag, "corr": c, "sig_band": 2 / np.sqrt(n)})
    return pd.DataFrame(rows).set_index("lag")


def block_bootstrap_mean_ci(x: pd.Series, block: int, n_boot: int = 1000,
                            ci: float = 0.90, seed: int = 42) -> tuple[float, float]:
    """Moving-block-bootstrap CI (in bps) for the mean of a return series.

    fwd_5d/fwd_20d observations overlap heavily (adjacent days share most of
    their forward window, and extreme-flow days cluster in runs), so
    resampling individual points as iid understates uncertainty. Resampling
    contiguous blocks instead preserves that dependence structure.
    """
    v = x.dropna().to_numpy()
    n = len(v)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[i] = v[idx].mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return (float(np.quantile(means, lo_q) * 1e4), float(np.quantile(means, hi_q) * 1e4))


def flow_quintile_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket days into flow-z quintiles; report mean forward returns with
    block-bootstrap 90% CIs so overlapping-window significance isn't overstated."""
    d = df.dropna(subset=["flow_z", "fwd_ret_1d"]).copy()
    d["q"] = pd.qcut(d["flow_z"], 5, labels=["Q1 big out", "Q2", "Q3", "Q4", "Q5 big in"])
    rows = []
    for q, grp in d.groupby("q", observed=True):
        lo5, hi5 = block_bootstrap_mean_ci(grp["fwd_ret_5d"], block=5)
        lo20, hi20 = block_bootstrap_mean_ci(grp["fwd_ret_20d"], block=20)
        rows.append({
            "q": q,
            "n": len(grp),
            "avg_flow_organic_$m": grp["net_flow_organic"].mean(),
            "avg_flow_total_$m": grp["net_flow"].mean(),
            "avg_flow_z": grp["flow_z"].mean(),
            "fwd_1d_bps": grp["fwd_ret_1d"].mean() * 1e4,
            "fwd_5d_bps": grp["fwd_ret_5d"].mean() * 1e4,
            "fwd_5d_ci90": f"[{lo5:+.0f}, {hi5:+.0f}]",
            "fwd_20d_bps": grp["fwd_ret_20d"].mean() * 1e4,
            "fwd_20d_ci90": f"[{lo20:+.0f}, {hi20:+.0f}]",
            "hit_1d": (grp["fwd_ret_1d"] > 0).mean(),
        })
    out = pd.DataFrame(rows).set_index("q")
    out.index.name = "q"
    for c in ("n",):
        out[c] = out[c].astype(int)
    return out.round(2)


def regime_gate(df: pd.DataFrame, slope_win: int = 20, ma_win: int = 50) -> pd.DataFrame:
    """Four-state regime ladder from 20d z-scored-cum-flow slope x price vs 50d SMA:
       LONG         : slope > 0, price > SMA  (grind up, short-put friendly)
       RECOVERY     : slope > 0, price < SMA  (flows turn first; bottoming)
       DISTRIBUTION : slope < 0, price > SMA  (price holds, flows leave; topping)
       STAND_DOWN   : slope < 0, price < SMA  (confirmed downtrend)

    Direction is decided on flow_slope_z (slope of cumsum(flow_z)), not raw
    dollars: raw cum-flow slope drifts with the ETF category's growing AUM
    base, which biases the sign toward whichever direction dollar flows
    have structurally trended lately. flow_slope (raw $m/day) is still
    computed and kept for reporting."""
    d = df.copy()
    d["cum_flow_z"] = d["flow_z"].fillna(0).cumsum()
    x = np.arange(slope_win)

    def slope(y: np.ndarray) -> float:
        return np.polyfit(x, y, 1)[0]

    d["flow_slope"] = d["cum_flow"].rolling(slope_win).apply(slope, raw=True)
    d["flow_slope_organic"] = d["cum_flow_organic"].rolling(slope_win).apply(slope, raw=True)
    d["flow_slope_z"] = d["cum_flow_z"].rolling(slope_win).apply(slope, raw=True)
    d["sma"] = d["btc_close"].rolling(ma_win).mean()
    up, above = d["flow_slope_z"] > 0, d["btc_close"] > d["sma"]
    conds = [up & above, up & ~above, ~up & above]
    d["regime"] = np.select(conds, ["LONG", "RECOVERY", "DISTRIBUTION"],
                            default="STAND_DOWN")
    d.loc[d["flow_slope_z"].isna() | d["sma"].isna(), "regime"] = "WARMUP"
    return d


def regime_stats(d: pd.DataFrame) -> pd.DataFrame:
    """Per-regime forward-return stats with block-bootstrap 90% CIs.
    Regime membership runs in multi-day streaks, so consecutive 1d returns
    within a regime aren't independent draws either — same overlap-inflation
    fix as the quintile table."""
    g = d[d["regime"] != "WARMUP"].dropna(subset=["fwd_ret_1d"])
    has_vol_premium = "iv_rv_spread" in d.columns
    has_funding = "funding_ann_pct" in d.columns
    rows = []
    for reg, grp in g.groupby("regime", observed=True):
        lo, hi = block_bootstrap_mean_ci(grp["fwd_ret_1d"], block=5)
        row = {
            "regime": reg,
            "days": len(grp),
            "avg_fwd_1d_bps": grp["fwd_ret_1d"].mean() * 1e4,
            "avg_fwd_1d_ci90": f"[{lo:+.0f}, {hi:+.0f}]",
            "ann_vol_pct": grp["fwd_ret_1d"].std() * np.sqrt(365) * 100,
            "hit": (grp["fwd_ret_1d"] > 0).mean(),
        }
        if has_vol_premium:
            row["avg_iv_rv_spread"] = grp["iv_rv_spread"].mean()
        if has_funding:
            row["avg_funding_ann_%"] = grp["funding_ann_pct"].mean()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("regime").round(2)
    order = [r for r in ["LONG", "RECOVERY", "DISTRIBUTION", "STAND_DOWN"]
             if r in out.index]
    return out.loc[order]


def _pctile_rank_last(window: np.ndarray) -> float:
    return (window <= window[-1]).mean()


def divergences(d: pd.DataFrame, win: int = 20, threshold: float = 0.6) -> pd.DataFrame:
    """Continuous divergence score instead of a binary new-high/new-low flag.

    div_score = price's percentile rank within its own `win`-day range minus
    cum-flow's percentile rank within its `win`-day range, in [-1, +1].
    +1 means price sits at the top of its recent range while flow sits at
    the bottom (fully unconfirmed strength); 0 means they're moving together.
    The old binary flag (price at an exact `win`-day high while flow isn't)
    was a special case of this at rank==1.0 — that made it flip noisily on
    borderline days. bearish_div/bullish_div are kept as threshold crossings
    of the continuous score for backward compatibility with the report/chart."""
    d = d.copy()
    px_rank = d["btc_close"].rolling(win).apply(_pctile_rank_last, raw=True)
    fl_rank = d["cum_flow"].rolling(win).apply(_pctile_rank_last, raw=True)
    d["div_score"] = px_rank - fl_rank
    d["bearish_div"] = d["div_score"] > threshold
    d["bullish_div"] = d["div_score"] < -threshold
    return d


# ----------------------------------------------------------------------------
# 3. Reporting
# ----------------------------------------------------------------------------

def make_chart(d: pd.DataFrame, xc: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_vol_panel = "dvol" in d.columns
    n_panels = 4 if has_vol_panel else 3
    ratios = [2, 1.4, 1, 1] if has_vol_panel else [2, 1.4, 1]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 11 + (2.5 if has_vol_panel else 0)), sharex=False,
        gridspec_kw={"height_ratios": ratios},
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        for s in ax.spines.values():
            s.set_color("#30363d")

    # Panel 1: price with regime shading
    ax = axes[0]
    ax.plot(d.index, d["btc_close"], color="#58a6ff", lw=1.4, label="BTC close")
    ax.plot(d.index, d["sma"], color="#8b949e", lw=0.9, ls="--", label="50d SMA")
    colors = {"LONG": "#1a7f37", "RECOVERY": "#1f6feb",
              "DISTRIBUTION": "#9e6a03", "STAND_DOWN": "#b62324"}
    for reg, col in colors.items():
        mask = d["regime"] == reg
        ax.fill_between(d.index, d["btc_close"].min(), d["btc_close"].max(),
                        where=mask, color=col, alpha=0.12, label=f"{reg} regime")
    bd = d[d["bearish_div"]]
    bu = d[d["bullish_div"]]
    ax.scatter(bd.index, bd["btc_close"], marker="v", color="#f85149", s=28,
               zorder=5, label="bearish divergence")
    ax.scatter(bu.index, bu["btc_close"], marker="^", color="#3fb950", s=28,
               zorder=5, label="bullish divergence")
    ax.set_title("BTC vs ETF flow regime", color="#e6edf3")
    ax.legend(loc="upper left", fontsize=8, facecolor="#161b22",
              labelcolor="#e6edf3", edgecolor="#30363d")

    # Panel 2: cumulative flow, decomposed into organic (ex-GBTC) vs GBTC legacy bleed
    ax = axes[1]
    ax.plot(d.index, d["cum_flow"] / 1000, color="#2ea043", lw=1.4, label="Total")
    ax.plot(d.index, d["cum_flow_organic"] / 1000, color="#58a6ff", lw=1.1, label="Organic (ex-GBTC)")
    ax.plot(d.index, d["cum_flow_gbtc"] / 1000, color="#f85149", lw=1.0, ls="--", label="GBTC")
    ax.axhline(0, color="#30363d", lw=0.8)
    ax.set_title("Cumulative ETF net flow (US$bn)", color="#e6edf3")
    ax.legend(loc="upper left", fontsize=8, facecolor="#161b22",
              labelcolor="#e6edf3", edgecolor="#30363d")

    # Panel 3: cross-correlation
    ax = axes[2]
    ax.bar(xc.index, xc["corr"], color=["#f85149" if l < 0 else "#58a6ff"
                                        for l in xc.index])
    ax.axhline(xc["sig_band"].iloc[0], color="#8b949e", ls=":", lw=0.8)
    ax.axhline(-xc["sig_band"].iloc[0], color="#8b949e", ls=":", lw=0.8)
    ax.set_title("corr(flow[t], return[t+lag]) — bars right of 0 = flow leading",
                 color="#e6edf3")
    ax.set_xlabel("lag (days)", color="#8b949e")

    # Panel 4 (if available): implied vs realized vol -> the actual premium-selling edge
    if has_vol_panel:
        ax = axes[3]
        ax.plot(d.index, d["dvol"], color="#d29922", lw=1.2, label="DVOL (implied)")
        ax.plot(d.index, d["realized_vol_20d"], color="#8b949e", lw=1.0, label="20d realized vol")
        ax.fill_between(d.index, d["realized_vol_20d"], d["dvol"],
                        where=d["dvol"] >= d["realized_vol_20d"],
                        color="#3fb950", alpha=0.15, label="premium (sell edge)")
        ax.fill_between(d.index, d["realized_vol_20d"], d["dvol"],
                        where=d["dvol"] < d["realized_vol_20d"],
                        color="#f85149", alpha=0.15, label="IV cheap vs RV")
        ax.set_title("Vol risk premium: implied (DVOL) vs 20d realized", color="#e6edf3")
        ax.legend(loc="upper left", fontsize=8, facecolor="#161b22",
                  labelcolor="#e6edf3", edgecolor="#30363d")

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def print_report(d: pd.DataFrame, xc: pd.DataFrame, quintiles: pd.DataFrame,
                 rstats: pd.DataFrame, funding_quintiles: pd.DataFrame | None = None) -> None:
    last = d.dropna(subset=["flow_slope_z", "sma"]).iloc[-1]
    peak_flow = d["cum_flow"].max()
    print("=" * 64)
    print(f"ETF FLOW SIGNAL REPORT  —  {d.index[-1].date()}")
    print("=" * 64)
    print(f"BTC close            : ${last['btc_close']:,.0f}")
    print(f"Cum flow             : ${last['cum_flow']/1000:,.1f}bn "
          f"({(last['cum_flow']/peak_flow - 1)*100:+.1f}% vs peak)")
    def _bn(x: float) -> str:
        return f"-${-x/1000:,.1f}bn" if x < 0 else f"${x/1000:,.1f}bn"

    print(f"  of which organic   : {_bn(last['cum_flow_organic'])} "
          f"(ex-GBTC: IBIT/FBTC/etc net creations)")
    print(f"  of which GBTC      : {_bn(last['cum_flow_gbtc'])} "
          f"(legacy trust, mostly structural bleed)")
    print(f"20d flow slope       : {last['flow_slope']:+.1f} $m/day (Total) "
          f"| {last['flow_slope_organic']:+.1f} $m/day (organic ex-GBTC)")
    print(f"                        {last['flow_slope_z']:+.3f} z/day "
          f"— regime direction uses this (organic, scale-free basis)")
    print(f"Flow z-score today   : {last['flow_z']:+.2f} "
          f"(organic flow vs its trailing 90d mean/std)")
    print(f"Price vs 50d SMA     : "
          f"{'ABOVE' if last['btc_close'] > last['sma'] else 'BELOW'} "
          f"(SMA ${last['sma']:,.0f})")
    print(f"Divergence score     : {last['div_score']:+.2f} "
          f"(price pctile - flow pctile, 20d window; |score|>{0.6:.1f} flags)")
    print(f"REGIME               : {last['regime']}")
    if "dvol" in d.columns or "funding_ann_pct" in d.columns:
        print("-" * 64)
        print("Options-market context:")
        if "dvol" in d.columns and pd.notna(last.get("dvol")):
            print(f"  DVOL (30d implied vol) : {last['dvol']:.1f}%")
            print(f"  20d realized vol        : {last['realized_vol_20d']:.1f}%")
            print(f"  Vol risk premium (IV-RV): {last['iv_rv_spread']:+.1f}pt "
                  f"({'premium looks worth selling' if last['iv_rv_spread'] > 0 else 'premium is CHEAP vs realized -- caution selling here'})")
        if "funding_ann_pct" in d.columns and pd.notna(last.get("funding_ann_pct")):
            fz = last.get("funding_z")
            crowd = ""
            if pd.notna(fz):
                if fz > 1.5:
                    crowd = " -- crowded long, squeeze/unwind risk"
                elif fz < -1.5:
                    crowd = " -- crowded short, squeeze-up risk"
            print(f"  Perp funding (ann.)     : {last['funding_ann_pct']:+.1f}% "
                  f"(z={fz:+.2f}){crowd}")
    recent = d.iloc[-10:]
    if recent["bullish_div"].any():
        print("NOTE: bullish divergence fired in the last 10 sessions")
    if recent["bearish_div"].any():
        print("NOTE: bearish divergence fired in the last 10 sessions")
    print("-" * 64)
    print("Cross-correlation (flow[t] vs return[t+lag]):")
    key_lags = xc.loc[[-3, -2, -1, 0, 1, 2, 3]]
    for lag, row in key_lags.iterrows():
        star = " *" if abs(row["corr"]) > row["sig_band"] else ""
        print(f"  lag {lag:+d}: {row['corr']:+.3f}{star}")
    print("  (* = outside ~2 s.e. band; negative lags = price leads flow)")
    print("-" * 64)
    print("Forward returns by flow quintile:")
    print(quintiles.to_string())
    if funding_quintiles is not None and not funding_quintiles.empty:
        print("-" * 64)
        print("Forward returns by perp-funding quintile (crowding, independent of flows):")
        print(funding_quintiles.to_string())
    print("-" * 64)
    print("Regime backtest (next-day returns while in each regime):")
    print(rstats.to_string())
    ann = rstats["avg_fwd_1d_bps"] / 1e4 * 365 * 100
    for reg in rstats.index:
        print(f"  {reg}: ~{ann[reg]:+.0f}% annualised drift, "
              f"{rstats.loc[reg, 'ann_vol_pct']:.0f}% vol")
    print("  (overlap/one-cycle caveats apply; use for sizing, not forecasts)")
    # Regime-flip watch: what the 20d z-slope would be if the oldest k days
    # rolled out of the window and were replaced by zero-flow days (matches
    # the z-scored basis that now decides the regime — a "zero flow" day is
    # itself above- or below-average depending on the recent 90d mean, which
    # is exactly the effect that makes flows "hold flat" look like recovery
    # after a bad stretch).
    last_mean, last_std = last["flow_roll_mean"], last["flow_roll_std"]
    hyp_zero_z = (0 - last_mean) / last_std if pd.notna(last_std) and last_std else 0.0
    tail_z = d["flow_z"].iloc[-20:].to_numpy()
    x = np.arange(20)
    flips_in = None
    for k in range(1, 21):
        hyp = np.concatenate([tail_z[k:], np.full(k, hyp_zero_z)])
        if np.polyfit(x, hyp.cumsum(), 1)[0] > 0:
            flips_in = k
            break
    if last["regime"] in ("STAND_DOWN", "DISTRIBUTION") and flips_in is not None:
        nxt = "RECOVERY" if last["btc_close"] < last["sma"] else "LONG"
        print(f"FLIP WATCH: flow slope turns positive in ~{flips_in} sessions "
              f"if flows just hold at zero from here -> would enter {nxt}")
    print("=" * 64)


def export_readable_csv(d: pd.DataFrame, path: str) -> None:
    """Human-readable companion to the raw daily CSV: $ / % formatted strings."""
    r = pd.DataFrame(index=d.index)
    r["net_flow_$m"] = d["net_flow"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["net_flow_organic_$m"] = d["net_flow_organic"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["net_flow_gbtc_$m"] = d["net_flow_gbtc"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["cum_flow_$m"] = d["cum_flow"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["cum_flow_organic_$m"] = d["cum_flow_organic"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["cum_flow_gbtc_$m"] = d["cum_flow_gbtc"].map(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    r["flow_z"] = d["flow_z"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    r["btc_close"] = d["btc_close"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    for col in ("log_ret", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"):
        r[f"{col}_%"] = (d[col] * 100).map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    r["flow_slope_$m/d"] = d["flow_slope"].map(lambda x: f"{x:+,.1f}" if pd.notna(x) else "")
    r["flow_slope_organic_$m/d"] = d["flow_slope_organic"].map(lambda x: f"{x:+,.1f}" if pd.notna(x) else "")
    r["flow_slope_z"] = d["flow_slope_z"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    r["sma"] = d["sma"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    r["regime"] = d["regime"]
    r["div_score"] = d["div_score"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    r["bearish_div"] = d["bearish_div"]
    r["bullish_div"] = d["bullish_div"]
    if "dvol" in d.columns:
        r["dvol_%"] = d["dvol"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        r["realized_vol_20d_%"] = d["realized_vol_20d"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        r["iv_rv_spread_pt"] = d["iv_rv_spread"].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "")
    if "funding_ann_pct" in d.columns:
        r["funding_ann_%"] = d["funding_ann_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
        r["funding_z"] = d["funding_z"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    r.to_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-csv", default=None,
                    help="optional path to your own daily BTC CSV (date,close)")
    ap.add_argument("--flows-html", default=None,
                    help="optional path to a saved Farside HTML file (offline mode)")
    ap.add_argument("--out-prefix", default="etf_flow",
                    help="prefix for output files")
    ap.add_argument("--no-options-data", action="store_true",
                    help="skip fetching Binance funding rate / Deribit DVOL")
    args = ap.parse_args()

    if args.flows_html:
        with open(args.flows_html, encoding="utf-8") as fh:
            flows = parse_farside_table(fh.read())
    else:
        flows = fetch_farside()
    print(f"Farside: {len(flows)} daily rows, "
          f"{flows.index[0].date()} -> {flows.index[-1].date()}")

    btc = fetch_btc(args.btc_csv)
    df = build_frame(flows, btc)

    # Farside sometimes posts today's date before the per-fund flow data is
    # actually filled in (Total shows a placeholder, funds still show '-').
    # Trim trailing not-yet-reported day(s) so "last" below is the last
    # COMPLETE trading day, not a half-populated placeholder row.
    trimmed = 0
    while len(df) and pd.isna(df["net_flow_organic"].iloc[-1]):
        df = df.iloc[:-1]
        trimmed += 1
    if trimmed:
        print(f"Note: trimmed {trimmed} not-yet-reported day(s) from Farside "
              f"(today's flow data usually posts the next NY morning)")

    funding = dvol = None
    if not args.no_options_data:
        try:
            funding = fetch_funding_rate()
            print(f"Binance funding rate: {len(funding)} daily rows")
        except Exception as e:
            print(f"WARNING: could not fetch Binance funding rate ({e}); continuing without it")
        try:
            dvol = fetch_dvol()
            print(f"Deribit DVOL: {len(dvol)} daily rows")
        except Exception as e:
            print(f"WARNING: could not fetch Deribit DVOL ({e}); continuing without it")
    df = add_options_data(df, funding, dvol)

    xc = cross_correlation(df)
    quintiles = flow_quintile_forward_returns(df)
    funding_quintiles = funding_quintile_forward_returns(df)
    d = regime_gate(df)
    d = divergences(d)
    rstats = regime_stats(d)

    d.to_csv(f"{args.out_prefix}_daily.csv")
    export_readable_csv(d, f"{args.out_prefix}_daily_readable.csv")
    make_chart(d, xc, f"{args.out_prefix}_signal.png")
    print_report(d, xc, quintiles, rstats, funding_quintiles)
    print(f"Saved: {args.out_prefix}_daily.csv, {args.out_prefix}_daily_readable.csv, "
          f"{args.out_prefix}_signal.png")


if __name__ == "__main__":
    main()
