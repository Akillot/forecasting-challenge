from __future__ import annotations
import pandas as pd


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
      """Aggregate VM-level readings into one row per timestamp."""
      valid = cpu.dropna(subset=["avg_cpu"])
      grouped = valid.groupby("timestamp")
      out = pd.DataFrame({
          "active_vm_count": grouped["vm_id"].nunique(),
          "aggregate_cpu_utilization": grouped["avg_cpu"].mean(),
      })
      return out.reset_index().sort_values("timestamp", ignore_index=True)


def split_train_holdout(targets: pd.DataFrame, holdout: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
      if len(targets) <= holdout:
           raise ValueError("not enough timestamps for the holdout")
      return targets.iloc[:-holdout], targets.iloc[-holdout:]
