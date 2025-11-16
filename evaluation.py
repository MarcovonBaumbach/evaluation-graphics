#!/usr/bin/env python3

import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
ENABLE_TICKER_COMPARISON = True  # expects column 'ticker' in Excel

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

def load_data(path):
    if not os.path.exists(path):
        logging.error(f"Missing file: {path}")
        sys.exit(1)

    df = pd.read_excel(path)

    required = {"closed time", "realized pnl", "closed value"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"Missing columns: {missing}")
        sys.exit(1)

    df["closed time"] = pd.to_datetime(df["closed time"], errors="coerce")
    df = df.dropna(subset=["closed time"]).sort_values("closed time").reset_index(drop=True)
    df["month"] = df["closed time"].dt.to_period("M").astype(str)

    df["success"] = df["realized pnl"] > 0
    df["win_amount"] = df["realized pnl"].clip(lower=0)
    df["loss_amount"] = (-df["realized pnl"]).clip(lower=0)

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

    expectancy = None
    if avg_win is not None and avg_loss is not None:
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    df["equity"] = df["realized pnl"].cumsum()
    df["cumulative_closed_value"] = df["closed value"].cumsum()
    if df["realized pnl"].sum() < 0:
        df["net_pnl"] = df["realized pnl"] + df["position fee"] + df["funding fees"]
    else:
        df["net_pnl"] = df["realized pnl"] - df["position fee"] - df["funding fees"]

    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["peak"] - df["equity"]
    max_dd = float(df["drawdown"].max()) if not df["drawdown"].empty else 0

    monthly = df.groupby("month").agg(
        closed_value=("closed value", "sum"),
        pnl=("realized pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        win_amount=("win_amount", "sum"),
        loss_amount=("loss_amount", "sum"),
    )

    monthly_counts = df.groupby("month").agg(
        Wins=("success", "sum"),
        Losses=("success", lambda s: (~s).sum()),
        Trades=("success", "size"),
    )

    monthly["cumulative_pnl"] = monthly["pnl"].cumsum()

    return {
        "df": df,
        "monthly": monthly,
        "monthly_counts": monthly_counts,
        "total_trades": total_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "total_win_usd": total_win_usd,
        "total_loss_usd": total_loss_usd,
        "max_dd": max_dd,
    }

# ==============================================
# COMPARISON (TICKERS / STRATEGIES)
# ==============================================

def compute_comparisons(df):
    if "ticker" not in df.columns:
        return None

    comp = df.groupby("ticker").agg(
        Trades=("realized pnl", "count"),
        TotalPnL=("realized pnl", "sum"),
        WinRate=("success", "mean"),
    )
    comp["WinRate"] = comp["WinRate"] * 100
    return comp

# ==============================================
# CHARTS 
# ============================================== 
# ==============================================

def create_charts(df, monthly, monthly_counts, outdir, comparison=None):
    ensure_dir(outdir)
    charts = {}

    # ----------------------------------------------------
    # 1) Equity curve
    # ----------------------------------------------------

    plt.figure(figsize=(12, 5))

    # Plot original equity (cumulative realized PnL)
    plt.plot(df["closed time"], df["equity"], linewidth=1.6, label="Equity (Realized PnL)", color="green")

    # Plot cumulative net PnL (after fees, including funding fees)
    plt.plot(df["closed time"], df["net_pnl"].cumsum(),
            linewidth=1.6, linestyle="--", label="Equity (Net PnL)", color="blue")

    plt.title("Equity Curve (Cumulative PnL)")
    plt.xlabel("Closed Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(alpha=.25)

    # Label the balance BEFORE the first trade of each month for both lines
    monthly_first = df.groupby(df["closed time"].dt.to_period("M")).head(1)
    net_cum = df["net_pnl"].cumsum()

    for _, row in monthly_first.iterrows():
        x = row["closed time"]
        equity_before = df["equity"].loc[row.name] - row["realized pnl"]
        net_before = net_cum.loc[row.name] - row["net_pnl"]

        plt.text(x, equity_before, f"{equity_before:.0f}", fontsize=8, va="bottom", color="green")
        plt.text(x, net_before, f"{net_before:.0f}", fontsize=8, va="bottom", color="blue")
        # Improve x-axis ticks: show only monthly ticks
        unique_months = monthly_first["closed time"]
        plt.xticks(unique_months, [d.strftime("%Y-%m") for d in unique_months], rotation=45)

    # Save chart
    eq_path = f"{outdir}/equity.png"
    plt.tight_layout()
    plt.savefig(eq_path, dpi=130)
    plt.close()

    charts["equity"] = eq_path

    # ----------------------------------------------------
    # 2) Drawdown chart
    # ----------------------------------------------------

    plt.figure(figsize=(12, 4))

    # Compute cumulative net PnL
    df["cumulative_net_pnl"] = df["net_pnl"].cumsum()

    # Drawdowns
    df["peak_equity"] = df["equity"].cummax()
    df["drawdown_equity"] = df["peak_equity"] - df["equity"]

    df["peak_net"] = df["cumulative_net_pnl"].cummax()
    df["drawdown_net"] = df["peak_net"] - df["cumulative_net_pnl"]

    # Plot drawdowns
    plt.plot(df["closed time"], df["drawdown_equity"], color="red", linewidth=1.4, label="Drawdown (Equity)")
    plt.plot(df["closed time"], df["drawdown_net"], color="blue", linestyle="--", linewidth=1.4, label="Drawdown (Net PnL)")

    plt.title("Drawdown Curve")
    plt.xlabel("Closed Time")
    plt.ylabel("Drawdown ($)")
    plt.legend()
    plt.grid(alpha=.25)

    # Label first trade of each month
    monthly_first = df.groupby(df["closed time"].dt.to_period("M")).head(1)
    for _, row in monthly_first.iterrows():
        x = row["closed time"]
        dd_equity = row["peak_equity"] - row["equity"]
        dd_net = df["drawdown_net"].loc[row.name]
        plt.text(x, dd_equity, f"{dd_equity:.0f}", fontsize=8, va="bottom", color="red")
        plt.text(x, dd_net, f"{dd_net:.0f}", fontsize=8, va="bottom", color="blue")

    # Improve x-axis ticks
    unique_months = monthly_first["closed time"]
    plt.xticks(unique_months, [d.strftime("%Y-%m") for d in unique_months], rotation=45)

    # Save chart
    dd_path = f"{outdir}/drawdown.png"
    plt.tight_layout()
    plt.savefig(dd_path, dpi=130)
    plt.close()
    charts["drawdown"] = dd_path

    # ----------------------------------------------------
    # 3) PnL Histogram 
    # ----------------------------------------------------

    plt.figure(figsize=(8, 4))
    plt.hist(df["realized pnl"], bins=30)
    plt.title("Realized PnL Distribution")
    plt.xlabel("PnL ($)")
    plt.ylabel("Count")
    plt.grid(alpha=.25)
    hist = f"{outdir}/hist.png"
    plt.tight_layout(); plt.savefig(hist, dpi=130); plt.close()
    charts["hist"] = hist

    # ----------------------------------------------------
    # 4) Win vs Loss Pie 
    # ----------------------------------------------------

    sizes = [int(monthly_counts["Wins"].sum()), int(monthly_counts["Losses"].sum())]
    labels = ["Wins", "Losses"]
    plt.figure(figsize=(5, 5))
    if sum(sizes) == 0:
        plt.pie([1], labels=["No trades"], autopct="%1.1f%%")
    else:
        plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Win vs Loss")
    pie = f"{outdir}/pie.png"
    plt.tight_layout(); plt.savefig(pie, dpi=130); plt.close()
    charts["pie"] = pie

    # ----------------------------------------------------
    # 4.1) Win vs Loss Pie (Net PnL)
    # ----------------------------------------------------
    net_wins = (df["net_pnl"] > 0).sum()
    net_losses = (df["net_pnl"] <= 0).sum()

    sizes = [net_wins, net_losses]
    labels = ["Wins", "Losses"]

    plt.figure(figsize=(5, 5))
    if sum(sizes) == 0:
        plt.pie([1], labels=["No trades"], autopct="%1.1f%%")
    else:
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["green", "red"])
    plt.title("Win vs Loss (Net PnL)")
    pie_net_path = f"{outdir}/pie_net_pnl.png"
    plt.tight_layout()
    plt.savefig(pie_net_path, dpi=130)
    plt.close()
    charts["pie_net_pnl"] = pie_net_path

    # ----------------------------------------------------
    # 5) Monthly Wins vs Losses Bar Chart
    # ----------------------------------------------------

    if not monthly_counts.empty:
        x = np.arange(len(monthly_counts.index))
        width = 0.35
        plt.figure(figsize=(12, 5))
        bars1 = plt.bar(x - width/2, monthly_counts["Wins"], width=width, label="Wins")
        bars2 = plt.bar(x + width/2, monthly_counts["Losses"], width=width, label="Losses")
        plt.xticks(x, monthly_counts.index, rotation=45)
        plt.title("Monthly Wins vs Losses")
        plt.legend(); plt.grid(axis="y", alpha=.25)
        for b in list(bars1) + list(bars2):
            h = b.get_height(); plt.text(b.get_x()+b.get_width()/2, h, str(int(h)), ha='center', va='bottom')
        mb = f"{outdir}/monthly_win_loss.png"
        plt.tight_layout(); plt.savefig(mb, dpi=130); plt.close()
        charts["monthly_bar"] = mb
    else:
        charts["monthly_bar"] = None

    # ----------------------------------------------------
    # 5.1) Monthly Wins vs Losses Bar Chart (Net PnL)
    # ----------------------------------------------------
    monthly_counts_net = df.groupby("month").agg(
        Wins_net=("net_pnl", lambda s: (s > 0).sum()),
        Losses_net=("net_pnl", lambda s: (s <= 0).sum())
    )

    x = np.arange(len(monthly_counts_net.index))
    width = 0.35
    plt.figure(figsize=(12, 5))
    bars1 = plt.bar(x - width/2, monthly_counts_net["Wins_net"], width=width, label="Wins")
    bars2 = plt.bar(x + width/2, monthly_counts_net["Losses_net"], width=width, label="Losses")
    plt.xticks(x, monthly_counts_net.index, rotation=45)
    plt.title("Monthly Wins vs Losses (Net PnL)")
    plt.legend()
    plt.grid(axis="y", alpha=.25)

    # Optional: Add value labels on top of bars
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, str(int(h)), ha='center', va='bottom')

    mb_net_path = f"{outdir}/monthly_win_loss_net.png"
    plt.tight_layout()
    plt.savefig(mb_net_path, dpi=130)
    plt.close()
    charts["monthly_bar_net"] = mb_net_path

    # ----------------------------------------------------
    # 6) Profit Factor per Month
    # ----------------------------------------------------

    pf = []
    for m in monthly.index:
        wins = monthly.loc[m, "win_amount"]
        losses = monthly.loc[m, "loss_amount"]
        pf.append(wins / losses if losses > 0 else np.nan)
    plt.figure(figsize=(12, 4))
    plt.bar(monthly.index, pf, color='steelblue')
    plt.title("Profit Factor per Month")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=.25)
    pf_path = f"{outdir}/profit_factor_monthly.png"
    plt.tight_layout(); plt.savefig(pf_path, dpi=130); plt.close()
    charts["pf_monthly"] = pf_path

    # ----------------------------------------------------
    # 6.1) Profit Factor per Month (Net PnL)
    # ----------------------------------------------------

    pf_net = []
    months = df["month"].unique()
    for m in months:
        monthly_df = df[df["month"] == m]
        wins = monthly_df.loc[monthly_df["net_pnl"] > 0, "net_pnl"].sum()
        losses = -monthly_df.loc[monthly_df["net_pnl"] < 0, "net_pnl"].sum()
        pf_net.append(wins / losses if losses > 0 else np.nan)

    plt.figure(figsize=(12, 4))
    plt.bar(months, pf_net, color="steelblue")
    plt.title("Profit Factor per Month (Net PnL)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=.25)

    pf_net_path = f"{outdir}/profit_factor_monthly_net.png"
    plt.tight_layout()
    plt.savefig(pf_net_path, dpi=130)
    plt.close()
    charts["pf_monthly_net"] = pf_net_path

    return charts

# ==============================================
# PDF GENERATION
# ==============================================

def build_pdf(metrics, charts, comparison, pdfname):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(name="title", parent=styles["Heading1"], alignment=1)
    h2 = styles["Heading2"]

    df = metrics["df"]
    total_net_wins = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    total_net_losses = -df.loc[df["net_pnl"] < 0, "net_pnl"].sum()
    total_net_profit_factor = (total_net_wins / total_net_losses) if total_net_losses > 0 else math.inf

    doc = SimpleDocTemplate(pdfname, pagesize=letter)
    story = []

    # Title
    story.append(Paragraph("Trading Performance Report", title))
    story.append(Spacer(1, 12))

    # -----------------------------------------------
    # Overview Table
    # -----------------------------------------------
    ov = metrics
    overview_rows = [
        ["Total Trades", ov["total_trades"]],
        ["Wins", ov["n_wins"]],
        ["Losses", ov["n_losses"]],
        ["Win Rate", f"{ov['win_rate']*100:.2f}%"],
        ["Profit Factor", fmt(ov["profit_factor"],3)],
        ["Net Profit Factor (excluding Fees)", fmt(total_net_profit_factor, 3)],
        ["Expectancy per trade", f"{fmt(ov['expectancy'])} $"],
        ["Closed Value Total", f"{fmt(ov['df']["closed value"].sum())} $"],
        ["Total Win Amount", f"{fmt(ov['total_win_usd'])} $"],
        ["Total Loss Amount", f"{fmt(ov['total_loss_usd'])} $"],
        ["Maximum Drawdown", f"{fmt(ov['max_dd'])} $"],
        ["Total Net PnL (no Fees)", f"{fmt(ov['df']['net_pnl'].sum())} $"],
    ]

    t = Table(overview_rows, colWidths=[240, 220])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))

    story.append(Paragraph("Overview", h2))
    story.append(t)
    story.append(Spacer(1, 18))

    # -----------------------------------------------
    # Monthly Summary Table
    # -----------------------------------------------
    story.append(Paragraph("Monthly Trading Summary", h2))

    m = metrics["monthly"]
    mc = metrics["monthly_counts"]

    rows = [["Month","Wins","Losses","Trades","Win $","Loss $","PnL","Net PnL","Cumulative PnL"]]
    for month in m.index:
        rows.append([
            month,
            int(mc.loc[month,"Wins"]) if month in mc.index else 0,
            int(mc.loc[month,"Losses"]) if month in mc.index else 0,
            int(mc.loc[month,"Trades"]) if month in mc.index else 0,
            fmt(m.loc[month,"win_amount"]) + " $",
            fmt(m.loc[month,"loss_amount"]) + " $",
            fmt(m.loc[month,"pnl"]) + " $",
            fmt(m.loc[month,"net_pnl"]) + " $",
            fmt(m.loc[month,"cumulative_pnl"]) + " $",
        ])

    mt = Table(rows, colWidths=[70,50,50,50,80,70,70,70])
    mt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))

    story.append(mt)
    story.append(Spacer(1, 18))

    # -----------------------------------------------
    # Ticker/Strategy Comparison Table
    # -----------------------------------------------
    if comparison is not None:
        story.append(Paragraph("Ticker / Strategy Comparison", h2))

        comp_rows = [["Ticker","Trades","Total PnL","Win Rate %"]]
        for tkr, row in comparison.iterrows():
            comp_rows.append([
                tkr,
                int(row["Trades"]),
                fmt(row["TotalPnL"]) + " $",
                f"{fmt(row['WinRate'])}%",
            ])

        ct = Table(comp_rows, colWidths=[100,100,100,100])
        ct.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))

        story.append(ct)
        story.append(Spacer(1, 18))

    # -----------------------------------------------
    # Charts (stacked under each other)
    # -----------------------------------------------
    story.append(Paragraph("Charts", h2))
    story.append(Spacer(1, 8))

    order = ["equity", "drawdown", "hist", "pie", "pie_net_pnl", "monthly_bar", "monthly_bar_net", "pf_monthly", "pf_monthly_net",]
    for key in order:
        f = charts.get(key)
        if f and os.path.exists(f):
            story.append(Image(f, width=500, height=300))
            story.append(Spacer(1, 16))

    doc.build(story)
    logging.info(f"PDF created: {pdfname}")

# ==============================================
# MAIN
# ==============================================

def main():
    logging.info("Loading data...")
    df = load_data(INPUT_FILE)

    metrics = compute_metrics(df)

    comparison = compute_comparisons(df) if ENABLE_TICKER_COMPARISON else None

    charts = create_charts(metrics["df"], metrics["monthly"], metrics["monthly_counts"], OUTDIR)

    build_pdf(metrics, charts, comparison, PDF_NAME)

    logging.info("All done!")

if __name__ == "__main__":
    main()
