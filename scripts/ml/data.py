"""
data.py
=======
Load and validate the master handoff table against the Data Handoff Contract
(schema.py). Enforces the Section 2.4 format conventions:

  * one row per test (replicates are separate rows sharing design_id),
  * fixed units per column (we check columns exist; units are the team's promise),
  * missing values are NaN, never 0,
  * CLIPPED sensor rows (flagged in `notes`) are excluded from training,
  * `data_source` lets us train with / without FEA-augmented rows.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd

import schema as S


def load_master(path: str | Path) -> pd.DataFrame:
    """Load the master table from CSV or Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate(df: pd.DataFrame, strict: bool = False) -> list[str]:
    """Check the table against the contract. Returns a list of problems.

    strict=True raises on any problem; otherwise problems are warned and returned.
    """
    problems: list[str] = []

    missing_cols = [c for c in S.ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"missing contract columns: {missing_cols}")

    # Categorical domains
    for col, allowed in [("data_source", S.DATA_SOURCES),
                         ("topology", S.TOPOLOGIES)]:
        if col in df.columns:
            bad = set(df[col].dropna().unique()) - set(allowed)
            if bad:
                problems.append(f"{col} has values outside the contract: {sorted(bad)}")
    if "impact_zone" in df.columns:
        bad = set(df["impact_zone"].dropna().unique()) - set(S.IMPACT_ZONES) - {""}
        if bad:
            problems.append(f"impact_zone has values outside the contract: {sorted(bad)}")

    # Replicate identity: (design_id, replicate_id, data_source) should be unique
    key = ["design_id", "replicate_id", "data_source"]
    if all(k in df.columns for k in key):
        dups = df.duplicated(subset=key).sum()
        if dups:
            problems.append(f"{dups} duplicate (design_id, replicate_id, data_source) rows")

    # Temperature sanity (should be ~23 C, just logged)
    if "temperature_C" in df.columns and df["temperature_C"].notna().any():
        t = df["temperature_C"].dropna()
        if (t < 5).any() or (t > 40).any():
            problems.append("temperature_C has values far from lab ~23C; check logging")

    if problems:
        msg = "Data Handoff Contract validation found issues:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return problems


def is_clipped(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows whose `notes` flag clipped/invalid sensor data."""
    if "notes" not in df.columns:
        return pd.Series(False, index=df.index)
    notes = df["notes"].fillna("").str.lower()
    return notes.str.contains("clip") | notes.str.contains("exclude")


def training_view(df: pd.DataFrame,
                  data_sources: list[str] | None = None,
                  include_fea: bool = False,
                  drop_clipped: bool = True) -> pd.DataFrame:
    """Return the rows eligible for model training.

    data_sources : restrict to these sources (e.g. ['oblique_impact']).
    include_fea   : if False, FEA-sourced rows are dropped (train without sim by default;
                    spec says only fold FEA in AFTER FEA-vs-experiment agreement is shown).
    drop_clipped  : drop rows flagged as clipped/invalid in `notes`.
    """
    out = df.copy()
    if drop_clipped:
        out = out[~is_clipped(out)]
    if not include_fea and "data_source" in out.columns:
        out = out[out["data_source"] != "fea"]
    if data_sources is not None:
        out = out[out["data_source"].isin(data_sources)]
    return out.reset_index(drop=True)


def labeled(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows with a non-missing value for `target` (NaN means 'not measured')."""
    return df[df[target].notna()].reset_index(drop=True)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    root = Path(__file__).resolve().parents[2]
    df = load_master(root / "data" / "raw" / "dummy_master.csv")
    probs = validate(df)
    print(f"Loaded {len(df)} rows. Validation problems: {len(probs)}")
    tv = training_view(df, data_sources=["oblique_impact"])
    print(f"Oblique-impact training rows (clipped excluded): {len(tv)} "
          f"across {tv['design_id'].nunique()} designs")
