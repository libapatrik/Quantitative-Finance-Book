import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_spx_options(ticker: str = "^SPX") -> tuple[float, pd.DataFrame]:
    """Download full option chain across all expirations. Returns (S0, raw_df)."""
    spx = yf.Ticker(ticker)
    S0 = (
        spx.info.get("regularMarketPrice")
        or spx.info.get("previousClose")
        or spx.fast_info["lastPrice"]
    )

    fetch_time = datetime.now()
    chains = []
    for exp in spx.options:
        chain = spx.option_chain(exp)
        calls = chain.calls.assign(optionType="call", expiration=exp)
        puts = chain.puts.assign(optionType="put", expiration=exp)
        chains.append(pd.concat([calls, puts]))

    df = pd.concat(chains, ignore_index=True)
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["T"] = (df["expiration"] - fetch_time).dt.total_seconds() / (365.25 * 86400)
    df["mid"] = (df["bid"] + df["ask"]) / 2

    return S0, df


def save_snapshot(S0: float, df: pd.DataFrame, output_dir: Path) -> Path:
    """Save option chain as CSV + JSON sidecar. Returns path to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"spx_options_{ts}.csv"
    meta_path = csv_path.with_suffix(".json")

    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({"S0": S0, "fetched_at": ts}))

    print(f"Snapshot saved: {csv_path}")
    return csv_path


def load_latest_snapshot(snapshot_dir: Path) -> tuple[float, pd.DataFrame]:
    """Load the most recent snapshot. Raises FileNotFoundError if none exist."""
    snapshots = sorted(snapshot_dir.glob("spx_options_*.csv"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")

    csv_path = snapshots[-1]
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    df = pd.read_csv(csv_path, parse_dates=["expiration"])

    print(f"Loaded snapshot: {csv_path.name}  (fetched {meta['fetched_at']})")
    return meta["S0"], df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot SPX option chain for SSVI calibration.")
    parser.add_argument("--ticker", default="^SPX")
    parser.add_argument("--output-dir", default="calibration/snapshots")
    args = parser.parse_args()

    S0, df = fetch_spx_options(args.ticker)
    save_snapshot(S0, df, Path(args.output_dir))
