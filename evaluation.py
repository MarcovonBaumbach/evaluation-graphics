#!/usr/bin/env python3

import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==============================================
# CONFIGURATION
# ==============================================

INPUT_FILE = "bitunix.xlsx"
OUTDIR = "charts"
PDF_NAME = "Trading_Report.pdf"
TRADES_PER_YEAR = 252

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==============================================
# HELPERS
# ==============================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_mean(series):
    if series.empty:
        return None
    return float(series.mean())


def fmt(v, n=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{n}f}"


# ==============================================
# LOADING + CLEANING
# ==============================================

def guess_open_time_column(df):
    candidates = [
        'open time', 'open_time', 'opened time', 'opened_time', 'created time', 'created_time',
        'entry time', 'entry_time', 'opened'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def load_data(path):
    if not os.path.exists(path):
        logging.error(f"Missing file: {path}")
        sys.exit(1)

    df = pd.read_excel(path)

    required = {"closed time", "realized pnl", "closed value", "opening time"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"Missing columns: {missing}")
        sys.exit(1)

    # parse times
    df["closed time"] = pd.to_datetime(df["closed time"], errors="coerce", utc=False).dt.tz_localize(None)
    df["opening time"] = pd.to_datetime(df["opening time"], errors="coerce", utc=False).dt.tz_localize(None)

    open_col = guess_open_time_column(df)
    if open_col:
        df[open_col] = pd.to_datetime(df[open_col], errors="coerce", utc=False).dt.tz_localize(None)(df[open_col], errors="coerce")
        df.rename(columns={open_col: 'open_time'}, inplace=True)
    else:
        df['open_time'] = pd.NaT

    # derive month
    df = df.dropna(subset=["closed time"]).sort_values("closed time").reset_index(drop=True)
    df["month"] = df["closed time"].dt.to_period("M").astype(str)

    # success and win/loss amounts
    df["success"] = df["realized pnl"] > 0
    df["win_amount"] = df["realized pnl"].clip(lower=0)
    df["loss_amount"] = (-df["realized pnl"]).clip(lower=0)

    # extract asset from futures column if ticker not present
    if 'ticker' not in df.columns:
        if 'futures' in df.columns:
            # expecting like "BTCUSDT long ..."
            df['asset'] = df['futures'].astype(str).str.split().str[0]
        else:
            # fallback: try to extract from 'symbol' or 'instrument'
            for alt in ['symbol', 'instrument', 'pair']:
                if alt in df.columns:
                    df['asset'] = df[alt].astype(str).str.split().str[0]
                    break
            else:
                df['asset'] = 'UNKNOWN'
    else:
        df['asset'] = df['ticker']

    # position size: prefer 'closed value' but accept synonyms
    if 'position size' not in df.columns:
        df['position_size'] = df['closed value']
    else:
        df['position_size'] = df['position size']

    # fees
    for f in ['position fee', 'position_fee', 'positionFee']:
        if f in df.columns:
            df['position fee'] = df[f]
            break
    else:
        df['position fee'] = df.get('position fee', 0.0)

    for f in ['funding fees', 'funding_fees', 'fundingFee']:
        if f in df.columns:
            df['funding fees'] = df[f]
            break
    else:
        df['funding fees'] = df.get('funding fees', 0.0)

    # optional volume column
    if 'volume' not in df.columns and 'qty' in df.columns:
        df['volume'] = df['qty']

    return df


# ==============================================
# METRICS
# ==============================================

def compute_metrics(df):
    total_trades = len(df)
    n_wins = int(df["success"].sum())
    n_losses = int((~df["success"]).sum())
    win_rate = n_wins / total_trades if total_trades else 0

    total_win_usd = float(df["win_amount"].sum())
    total_loss_usd = float(df["loss_amount"].sum())

    profit_factor = (total_win_usd / total_loss_usd) if total_loss_usd > 0 else math.inf

    avg_win = safe_mean(df[df["realized pnl"] > 0]["realized pnl"])
    avg_loss = safe_mean(df[df["realized pnl"] < 0]["realized pnl"])  # negative

    # payoff ratio (avg win / abs(avg loss))
    payoff = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff = abs(avg_win / avg_loss)

    # expectancy (already implemented) and normalized expectancy
    expectancy = None
    expectancy_pct = None
    if avg_win is not None and avg_loss is not None:
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        avg_trade_size = df['position_size'].replace(0, np.nan).mean()
        if not np.isnan(avg_trade_size) and avg_trade_size != 0:
            expectancy_pct = expectancy / avg_trade_size

    # stddev of pnl per trade
    pnl_std = float(df['realized pnl'].std()) if not df['realized pnl'].empty else 0.0

    # equity / cumulative
    df['equity'] = df['realized pnl'].cumsum()
    df['cumulative_closed_value'] = df['closed value'].cumsum()

    # net pnl accounting for fees
    if df["realized pnl"].sum() < 0:
        df["net_pnl"] = df["realized pnl"] + df["position fee"] + df["funding fees"]
    else:
        df["net_pnl"] = df["realized pnl"] - df["position fee"] - df["funding fees"]

    # peak and drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = df['peak'] - df['equity']
    max_dd = float(df['drawdown'].max()) if not df['drawdown'].empty else 0

    # drawdown duration analysis (identify drawdown episodes)
    dd_episodes = []
    in_dd = False
    dd_start_idx = None
    peak_at_start = None
    for idx, row in df.iterrows():
        if row['equity'] < row['peak']:
            if not in_dd:
                in_dd = True
                dd_start_idx = idx
                peak_at_start = row['peak']
        else:
            if in_dd:
                in_dd = False
                dd_end_idx = idx
                dd_episodes.append((dd_start_idx, dd_end_idx, peak_at_start - df.loc[dd_end_idx, 'equity']))
    # if ends in dd
    if in_dd:
        dd_episodes.append((dd_start_idx, df.index[-1], peak_at_start - df.iloc[-1]['equity']))

    dd_durations = []
    for s, e, val in dd_episodes:
        start_time = df.loc[s, 'closed time']
        end_time = df.loc[e, 'closed time']
        dd_durations.append((end_time - start_time).total_seconds() / 86400.0)  # days

    longest_dd_duration = max(dd_durations) if dd_durations else 0
    avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0
    num_dd_per_month = len(dd_episodes) / max(1, df['month'].nunique())

    # trade duration (if open_time present)
    if 'opening time' in df.columns:
        df['duration'] = (df['closed time'] - df['opening time']).dt.total_seconds()
        df['duration_minutes'] = df['duration'] / 60.0
        df['duration_hours'] = df['duration'] / 3600.0
    else:
        df['duration_minutes'] = np.nan
        df['duration_hours'] = np.nan


    # streaks
    streaks = []
    cur_type = None
    cur_len = 0
    best_win_streak = 0
    best_loss_streak = 0
    for win in df['success']:
        if cur_type is None:
            cur_type = win
            cur_len = 1
        else:
            if win == cur_type:
                cur_len += 1
            else:
                # flush
                if cur_type:
                    best_win_streak = max(best_win_streak, cur_len)
                else:
                    best_loss_streak = max(best_loss_streak, cur_len)
                cur_type = win
                cur_len = 1
    # final flush
    if cur_type:
        best_win_streak = max(best_win_streak, cur_len)
    else:
        best_loss_streak = max(best_loss_streak, cur_len)

    # position size analytics
    avg_position_size = float(df['position_size'].mean()) if not df['position_size'].empty else 0
    avg_size_wins = float(df.loc[df['realized pnl'] > 0, 'position_size'].mean()) if not df.loc[df['realized pnl'] > 0].empty else 0
    avg_size_losses = float(df.loc[df['realized pnl'] < 0, 'position_size'].mean()) if not df.loc[df['realized pnl'] < 0].empty else 0
    corr_size_pnl = float(df['position_size'].corr(df['realized pnl'])) if total_trades > 1 else 0

    # fee analysis
    total_fees = float(df.get('position fee', 0.0).sum() + df.get('funding fees', 0.0).sum())
    gross_pnl = float(df['realized pnl'].sum())
    fee_ratio = total_fees / gross_pnl if gross_pnl != 0 else np.nan

    # per-asset comparisons
    per_asset = df.groupby('asset').agg(
        Trades=('realized pnl', 'count'),
        TotalPnL=('realized pnl', 'sum'),
        WinRate=('success', 'mean'),
        AvgWin=('win_amount', 'mean'),
        AvgLoss=('loss_amount', 'mean'),
        AvgPositionSize=('position_size', 'mean')
    )
    per_asset['WinRate'] = per_asset['WinRate'] * 100
    per_asset['ProfitFactor'] = per_asset['AvgWin'] * per_asset['Trades'] / (per_asset['AvgLoss'] * per_asset['Trades']).replace(0, np.nan)

    # outliers
    pnl_z = (df['realized pnl'] - df['realized pnl'].mean()) / (df['realized pnl'].std() if df['realized pnl'].std() != 0 else 1)
    df['pnl_z'] = pnl_z
    outliers = df.loc[pnl_z.abs() > 3].sort_values('pnl_z', key=abs, ascending=False)

    # duration buckets for analysis
    bins = [0, 2*60, 10*60, 30*60, 60*60, 4*3600, 24*3600]  # seconds
    labels = ['0-2m', '2-10m', '10-30m', '30-60m', '1-4h', '4h-24h']
    try:
        df['duration_bucket'] = pd.cut(df['duration'].fillna(0), bins=bins, labels=labels, include_lowest=True)
    except Exception:
        df['duration_bucket'] = np.nan

    # time of day
    df['hour'] = df['closed time'].dt.hour
    hourly = df.groupby('hour').agg(
        PnL=('realized pnl', 'sum'),
        Trades=('realized pnl', 'count'),
        WinRate=('success', 'mean')
    )
    hourly['WinRate'] = hourly['WinRate'] * 100

    df['entry_hour'] = df['opening time'].dt.hour
    hourly_entry = df.groupby('entry_hour').agg(
        PnL=('realized pnl', 'sum'),
        Trades=('realized pnl', 'count'),
        WinRate=('success', 'mean')
    )

    # monthly summaries
    monthly = df.groupby('month').agg(
        closed_value=('closed value', 'sum'),
        pnl=('realized pnl', 'sum'),
        net_pnl=('net_pnl', 'sum'),
        win_amount=('win_amount', 'sum'),
        loss_amount=('loss_amount', 'sum'),
    )

    monthly_counts = df.groupby('month').agg(
        Wins=('success', 'sum'),
        Losses=('success', lambda s: (~s).sum()),
        Trades=('success', 'size'),
    )

    monthly['cumulative_pnl'] = monthly['pnl'].cumsum()

    return {
        'df': df,
        'monthly': monthly,
        'monthly_counts': monthly_counts,
        'total_trades': total_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'expectancy_pct': expectancy_pct,
        'payoff': payoff,
        'pnl_std': pnl_std,
        'total_win_usd': total_win_usd,
        'total_loss_usd': total_loss_usd,
        'max_dd': max_dd,
        'longest_dd_duration': longest_dd_duration,
        'avg_dd_duration': avg_dd_duration,
        'num_dd_per_month': num_dd_per_month,
        'best_win_streak': best_win_streak,
        'best_loss_streak': best_loss_streak,
        'avg_position_size': avg_position_size,
        'avg_size_wins': avg_size_wins,
        'avg_size_losses': avg_size_losses,
        'corr_size_pnl': corr_size_pnl,
        'total_fees': total_fees,
        'fee_ratio': fee_ratio,
        'per_asset': per_asset,
        'hourly': hourly,
        'hourly_entry': hourly_entry,
        'outliers': outliers,
    }


# ==============================================
# CHARTS
# ==============================================

def create_charts(df, monthly, monthly_counts, outdir, metrics):
    ensure_dir(outdir)
    charts = {}

    # 1) Equity curve
    plt.figure(figsize=(12, 5))
    plt.plot(df['closed time'], df['equity'], linewidth=1.6, label='Equity (Realized PnL)', color='green')
    plt.plot(df['closed time'], df['net_pnl'].cumsum(), linewidth=1.6, linestyle='--', label='Equity (Net PnL)', color='blue')
    plt.title('Equity Curve (Cumulative PnL)')
    plt.xlabel('Closed Time')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(alpha=.25)
    monthly_first = df.groupby(df['closed time'].dt.to_period('M')).head(1)
    net_cum = df['net_pnl'].cumsum()
    for _, row in monthly_first.iterrows():
        x = row['closed time']
        equity_before = df['equity'].loc[row.name] - row['realized pnl']
        net_before = net_cum.loc[row.name] - row['net_pnl']
        plt.text(x, equity_before, f"{equity_before:.0f}", fontsize=8, va='bottom', color='green')
        plt.text(x, net_before, f"{net_before:.0f}", fontsize=8, va='bottom', color='blue')
        unique_months = monthly_first['closed time']
        plt.xticks(unique_months, [d.strftime('%Y-%m') for d in unique_months], rotation=45)
    eq_path = f"{outdir}/equity.png"
    plt.tight_layout(); plt.savefig(eq_path, dpi=130); plt.close()
    charts['equity'] = eq_path

    # 2) Drawdown chart
    plt.figure(figsize=(12, 4))
    df['cumulative_net_pnl'] = df['net_pnl'].cumsum()
    df['peak_equity'] = df['equity'].cummax()
    df['drawdown_equity'] = df['peak_equity'] - df['equity']
    df['peak_net'] = df['cumulative_net_pnl'].cummax()
    df['drawdown_net'] = df['peak_net'] - df['cumulative_net_pnl']
    plt.plot(df['closed time'], df['drawdown_equity'], color='red', linewidth=1.4, label='Drawdown (Equity)')
    plt.plot(df['closed time'], df['drawdown_net'], color='blue', linestyle='--', linewidth=1.4, label='Drawdown (Net PnL)')
    plt.title('Drawdown Curve')
    plt.xlabel('Closed Time')
    plt.ylabel('Drawdown ($)')
    plt.legend(); plt.grid(alpha=.25)
    monthly_first = df.groupby(df['closed time'].dt.to_period('M')).head(1)
    for _, row in monthly_first.iterrows():
        x = row['closed time']
        dd_equity = row['peak_equity'] - row['equity']
        dd_net = df['drawdown_net'].loc[row.name]
        plt.text(x, dd_equity, f"{dd_equity:.0f}", fontsize=8, va='bottom', color='red')
        plt.text(x, dd_net, f"{dd_net:.0f}", fontsize=8, va='bottom', color='blue')
    unique_months = monthly_first['closed time']
    plt.xticks(unique_months, [d.strftime('%Y-%m') for d in unique_months], rotation=45)
    dd_path = f"{outdir}/drawdown.png"
    plt.tight_layout(); plt.savefig(dd_path, dpi=130); plt.close()
    charts['drawdown'] = dd_path

    # 3) PnL Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(df['realized pnl'], bins=30)
    plt.title('Realized PnL Distribution')
    plt.xlabel('PnL ($)')
    plt.ylabel('Count')
    plt.grid(alpha=.25)
    hist = f"{outdir}/hist.png"
    plt.tight_layout(); plt.savefig(hist, dpi=130); plt.close()
    charts['hist'] = hist

    # 4) Win vs Loss Pie
    sizes = [int(monthly_counts['Wins'].sum()), int(monthly_counts['Losses'].sum())]
    labels = ['Wins', 'Losses']
    plt.figure(figsize=(5, 5))
    if sum(sizes) == 0:
        plt.pie([1], labels=['No trades'], autopct='%1.1f%%')
    else:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Win vs Loss')
    pie = f"{outdir}/pie.png"
    plt.tight_layout(); plt.savefig(pie, dpi=130); plt.close()
    charts['pie'] = pie

    # 4.1) Win vs Loss Pie (Net PnL)
    net_wins = (df['net_pnl'] > 0).sum()
    net_losses = (df['net_pnl'] <= 0).sum()
    sizes = [net_wins, net_losses]
    labels = ['Wins', 'Losses']
    plt.figure(figsize=(5, 5))
    if sum(sizes) == 0:
        plt.pie([1], labels=['No trades'], autopct='%1.1f%%')
    else:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Win vs Loss (Net PnL)')
    pie_net_path = f"{outdir}/pie_net_pnl.png"
    plt.tight_layout(); plt.savefig(pie_net_path, dpi=130); plt.close()
    charts['pie_net_pnl'] = pie_net_path

    # 5) Monthly Wins vs Losses Bar Chart
    if not monthly_counts.empty:
        x = np.arange(len(monthly_counts.index))
        width = 0.35
        plt.figure(figsize=(12, 5))
        bars1 = plt.bar(x - width/2, monthly_counts['Wins'], width=width, label='Wins')
        bars2 = plt.bar(x + width/2, monthly_counts['Losses'], width=width, label='Losses')
        plt.xticks(x, monthly_counts.index, rotation=45)
        plt.title('Monthly Wins vs Losses')
        plt.legend(); plt.grid(axis='y', alpha=.25)
        for b in list(bars1) + list(bars2):
            h = b.get_height(); plt.text(b.get_x()+b.get_width()/2, h, str(int(h)), ha='center', va='bottom')
        mb = f"{outdir}/monthly_win_loss.png"
        plt.tight_layout(); plt.savefig(mb, dpi=130); plt.close()
        charts['monthly_bar'] = mb
    else:
        charts['monthly_bar'] = None

    # 5.1) Monthly Wins vs Losses (Net PnL)
    monthly_counts_net = df.groupby('month').agg(
        Wins_net=('net_pnl', lambda s: (s > 0).sum()),
        Losses_net=('net_pnl', lambda s: (s <= 0).sum())
    )
    x = np.arange(len(monthly_counts_net.index))
    width = 0.35
    plt.figure(figsize=(12, 5))
    bars1 = plt.bar(x - width/2, monthly_counts_net['Wins_net'], width=width, label='Wins')
    bars2 = plt.bar(x + width/2, monthly_counts_net['Losses_net'], width=width, label='Losses')
    plt.xticks(x, monthly_counts_net.index, rotation=45)
    plt.title('Monthly Wins vs Losses (Net PnL)')
    plt.legend(); plt.grid(axis='y', alpha=.25)
    for b in list(bars1) + list(bars2):
        h = b.get_height(); plt.text(b.get_x()+b.get_width()/2, h, str(int(h)), ha='center', va='bottom')
    mb_net_path = f"{outdir}/monthly_win_loss_net.png"
    plt.tight_layout(); plt.savefig(mb_net_path, dpi=130); plt.close()
    charts['monthly_bar_net'] = mb_net_path

    # 6) Profit Factor per Month
    pf = []
    for m in monthly.index:
        wins = monthly.loc[m, 'win_amount']
        losses = monthly.loc[m, 'loss_amount']
        pf.append(wins / losses if losses > 0 else np.nan)
    plt.figure(figsize=(12, 4))
    plt.bar(monthly.index, pf)
    plt.title('Profit Factor per Month')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=.25)
    pf_path = f"{outdir}/profit_factor_monthly.png"
    plt.tight_layout(); plt.savefig(pf_path, dpi=130); plt.close()
    charts['pf_monthly'] = pf_path

    # 6.1) Profit Factor per Month (Net PnL)
    pf_net = []
    months = df['month'].unique()
    for m in months:
        monthly_df = df[df['month'] == m]
        wins = monthly_df.loc[monthly_df['net_pnl'] > 0, 'net_pnl'].sum()
        losses = -monthly_df.loc[monthly_df['net_pnl'] < 0, 'net_pnl'].sum()
        pf_net.append(wins / losses if losses > 0 else np.nan)
    plt.figure(figsize=(12, 4))
    plt.bar(months, pf_net)
    plt.title('Profit Factor per Month (Net PnL)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=.25)
    pf_net_path = f"{outdir}/profit_factor_monthly_net.png"
    plt.tight_layout(); plt.savefig(pf_net_path, dpi=130); plt.close()
    charts['pf_monthly_net'] = pf_net_path

    # 7) Hourly performance bar
    try:
        h = metrics['hourly_entry']
        plt.figure(figsize=(10,4))
        plt.bar(h.index, h['PnL'])
        plt.title('PnL by Hour of Day (when trades have been opened)')
        plt.xlabel('h')
        plt.ylabel('PnL')
        plt.grid(axis='y', alpha=.25)
        hourly_path = f"{outdir}/pnl_by_hour.png"
        plt.tight_layout(); plt.savefig(hourly_path, dpi=130); plt.close()
        charts['pnl_by_hour'] = hourly_path
    except Exception:
        charts['pnl_by_hour'] = None

    # 8) Duration histogram
    try:
        plt.figure(figsize=(8,4))
        plt.hist(df['duration_hours'].dropna(), bins=30)
        plt.title('Trade Duration')
        plt.xlabel('Hours')
        plt.ylabel('Count')
        plt.grid(alpha=.25)
        dur_path = f"{outdir}/duration_hist.png"
        plt.tight_layout(); plt.savefig(dur_path, dpi=130); plt.close()
        charts['duration_hist'] = dur_path
    except Exception:
        charts['duration_hist'] = None


    # 9) PnL vs Duration scatter
    try:
        plt.figure(figsize=(8,4))
        plt.scatter(df['duration_hours'], df['realized pnl'], s=10, alpha=0.6)
        plt.title('PnL vs Trade Duration')
        plt.xlabel('Duration (hours)')
        plt.ylabel('Realized PnL ($)')
        plt.grid(alpha=.25)
        scatter_path = f"{outdir}/pnl_vs_duration.png"
        plt.tight_layout(); plt.savefig(scatter_path, dpi=130); plt.close()
        charts['pnl_vs_duration'] = scatter_path
    except Exception:
        charts['pnl_vs_duration'] = None

    # 10) Drawdown durations bar
    try:
        plt.figure(figsize=(10,4))
        durations = [d for d in metrics['df']['drawdown'] if not pd.isna(d)]
        # reuse earlier computed longest/avg
        stats_txt = f"Longest DD (days): {metrics['longest_dd_duration']:.1f}\nAvg DD (days): {metrics['avg_dd_duration']:.1f}"
        plt.text(0.05, 0.5, stats_txt, fontsize=12)
        plt.title('Drawdown Summary (text)')
        ddtext_path = f"{outdir}/drawdown_summary.png"
        plt.axis('off')
        plt.savefig(ddtext_path, dpi=130, bbox_inches='tight'); plt.close()
        charts['drawdown_summary'] = ddtext_path
    except Exception:
        charts['drawdown_summary'] = None

    # 11) Position size hist
    try:
        plt.figure(figsize=(8,4))
        plt.hist(df['position_size'].dropna(), bins=30)
        plt.title('Position Size Distribution')
        plt.xlabel('Closed Value (position size)')
        plt.ylabel('Count')
        plt.grid(alpha=.25)
        size_path = f"{outdir}/position_size_hist.png"
        plt.tight_layout(); plt.savefig(size_path, dpi=130); plt.close()
        charts['position_size_hist'] = size_path
    except Exception:
        charts['position_size_hist'] = None

    # 12) Per-asset PnL bar
    try:
        pa = metrics['per_asset'].sort_values('TotalPnL', ascending=False).head(20)
        plt.figure(figsize=(10,4))
        plt.bar(pa.index.astype(str), pa['TotalPnL'])
        plt.xticks(rotation=45)
        plt.title('PnL by Asset (top 20)')
        plt.grid(axis='y', alpha=.25)
        asset_path = f"{outdir}/pnl_by_asset.png"
        plt.tight_layout(); plt.savefig(asset_path, dpi=130); plt.close()
        charts['pnl_by_asset'] = asset_path
    except Exception:
        charts['pnl_by_asset'] = None

    # 13) Fee impact chart (monthly)
    try:
        monthly_fees = df.groupby('month').agg(total_fees=('position fee', 'sum'), funding=('funding fees', 'sum'))
        monthly_fees['fees'] = monthly_fees['total_fees'] + monthly_fees['funding']
        plt.figure(figsize=(10,4))
        plt.bar(monthly_fees.index, monthly_fees['fees'])
        plt.title('Fees Paid per Month')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=.25)
        fees_path = f"{outdir}/fees_per_month.png"
        plt.tight_layout(); plt.savefig(fees_path, dpi=130); plt.close()
        charts['fees_per_month'] = fees_path
    except Exception:
        charts['fees_per_month'] = None

    return charts


# ==============================================
# PDF GENERATION
# ==============================================

def build_pdf(metrics, charts, pdfname):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(name='title', parent=styles['Heading1'], alignment=1)
    h2 = styles['Heading2']

    df = metrics['df']

    doc = SimpleDocTemplate(pdfname, pagesize=letter)
    story = []

    story.append(Paragraph('Trading Performance Report', title))
    story.append(Spacer(1,12))

    # Overview Table
    ov = metrics
    overview_rows = [
        ['Total Trades', ov['total_trades']],
        ['Wins', ov['n_wins']],
        ['Losses', ov['n_losses']],
        ['Win Rate', f"{ov['win_rate']*100:.2f}%"],
        ['Profit Factor', fmt(ov['profit_factor'],3)],
        ['Net Profit Factor (based on net_pnl)', fmt((lambda df: (df.loc[df['net_pnl']>0,'net_pnl'].sum() / (-df.loc[df['net_pnl']<0,'net_pnl'].sum())) if (-df.loc[df['net_pnl']<0,'net_pnl'].sum())>0 else math.inf)(ov['df']),3)],
        ['Expectancy per trade', f"{fmt(ov['expectancy'])} $"],
        ['Expectancy (% of avg pos)', f"{fmt(ov.get('expectancy_pct', None),3)}"],
        ['Payoff Ratio (avgW/avgL)', fmt(ov.get('payoff', None),3)],
        ['PnL Std Dev (per trade)', f"{fmt(ov.get('pnl_std', None))} $"],
        ['Average Position Size', f"{fmt(ov.get('avg_position_size', None))} $"],
        ['Total Fees Paid', f"{fmt(ov.get('total_fees', 0.0))} $"],
        ['Fee ratio (fees/gross pnl)', f"{fmt(ov.get('fee_ratio', None),3)}"],
        ['Maximum Drawdown', f"{fmt(ov['max_dd'])} $"],
        ['Longest Drawdown (days)', f"{fmt(ov.get('longest_dd_duration',0),2)}"],
        ['Closed Value Total', f"{fmt(ov['df']['closed value'].sum())} $"],
    ]

    t = Table(overview_rows, colWidths=[260, 200])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.4, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(Paragraph('Overview', h2))
    story.append(t)
    story.append(Spacer(1,18))

    # Monthly Summary
    story.append(Paragraph('Monthly Trading Summary', h2))
    m = metrics['monthly']
    mc = metrics['monthly_counts']
    rows = [['Month','Wins','Losses','Trades','Win $','Loss $','PnL','Net PnL','Cumulative PnL']]
    for month in m.index:
        rows.append([
            month,
            int(mc.loc[month,'Wins']) if month in mc.index else 0,
            int(mc.loc[month,'Losses']) if month in mc.index else 0,
            int(mc.loc[month,'Trades']) if month in mc.index else 0,
            fmt(m.loc[month,'win_amount']) + ' $',
            fmt(m.loc[month,'loss_amount']) + ' $',
            fmt(m.loc[month,'pnl']) + ' $',
            fmt(m.loc[month,'net_pnl']) + ' $',
            fmt(m.loc[month,'cumulative_pnl']) + ' $',
        ])
    mt = Table(rows, colWidths=[70,50,50,50,80,70,70,70])
    mt.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1,18))

    # Per-asset summary table
    story.append(Paragraph('Per-Asset Summary (Top assets)', h2))
    pa = metrics['per_asset'].sort_values('TotalPnL', ascending=False).head(20)
    parows = [['Asset','Trades','Total PnL','Win Rate (%)','Avg Pos Size']]
    for idx, r in pa.iterrows():
        parows.append([str(idx), int(r['Trades']), fmt(r['TotalPnL']), fmt(r['WinRate'],2), fmt(r['AvgPositionSize'])])
    pat = Table(parows, colWidths=[100,60,80,80,100])
    pat.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(pat)
    story.append(Spacer(1,18))

    # Streaks + Outliers summary
    story.append(Paragraph('Streaks & Outliers', h2))
    streak_rows = [
        ['Best Win Streak', metrics['best_win_streak']],
        ['Best Loss Streak', metrics['best_loss_streak']],
        ['Top 5 Wins (amount)', ', '.join([fmt(x) for x in sorted(metrics['df']['realized pnl'], reverse=True)[:5]])],
        ['Top 5 Losses (amount)', ', '.join([fmt(x) for x in sorted(metrics['df']['realized pnl'])[:5]])],
    ]
    st = Table(streak_rows, colWidths=[240, 220])
    st.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(st)
    story.append(Spacer(1,18))

    # Charts
    story.append(Paragraph('Charts', h2)); story.append(Spacer(1,8))
    order = [
        'equity','drawdown','hist','pie','pie_net_pnl','monthly_bar','monthly_bar_net','pf_monthly','pf_monthly_net',
        'pnl_by_hour', 'duration_hist', 'pnl_vs_duration', 'position_size_hist','pnl_by_asset','fees_per_month','outliers','drawdown_summary'
    ]
    for key in order:
        f = charts.get(key)
        if f and os.path.exists(f):
            story.append(Image(f, width=500, height=300))
            story.append(Spacer(1,16))

    # Legend / Explanations
    story.append(Paragraph('Legend (metric explanations)', h2))
    legend_rows = [
        ['Metric','Explanation'],
        ['Total Trades','Total number of closed trades processed from the file.'],
        ['Wins / Losses','Count of trades with positive / non-positive realized PnL.'],
        ['Win Rate','Wins divided by Total Trades (percentage).'],
        ['Profit Factor','Gross wins divided by gross losses (using realized pnl).'],
        ['Net Profit Factor (based on net_pnl)','Profit factor computed using net_pnl (realized pnl minus fees).'],
        ['Expectancy per trade','Average expected value per trade in USD: WinRate*AvgWin + (1-WinRate)*AvgLoss.'],
        ['Payoff Ratio','Average win divided by average loss (absolute value).'],
        ['PnL Std Dev','Standard deviation of realized PnL per trade (volatility of trade results).'],
        ['Average Position Size','Mean of closed value (used as proxy for position size).'],
        ['Fee ratio','Total fees divided by gross PnL (shows fees impact).'],
        ['Maximum Drawdown','Largest peak-to-trough drop in cumulative realized PnL.'],
        ['Longest Drawdown (days)','Longest duration (in days) spent underwater between peak and recovery.'],
    ]
    lt = Table(legend_rows, colWidths=[160,340])
    lt.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(lt)
    story.append(Spacer(1,12))

    doc.build(story)
    logging.info(f"PDF created: {pdfname}")


# ==============================================
# MAIN
# ==============================================

def main():
    logging.info('Loading data...')
    df = load_data(INPUT_FILE)

    metrics = compute_metrics(df)

    charts = create_charts(metrics['df'], metrics['monthly'], metrics['monthly_counts'], OUTDIR, metrics)

    build_pdf(metrics, charts, PDF_NAME)

    logging.info('All done!')

if __name__ == '__main__':
    main()
