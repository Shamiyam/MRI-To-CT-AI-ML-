#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import numpy as np


def filter_by_modality(df: pd.DataFrame, source_modality: str) -> pd.DataFrame:
    sm = source_modality.lower()
    if sm == "all":
        return df.copy()
    return df[df["source_modality"].str.lower() == sm].copy()


def stratified_split(df: pd.DataFrame, seed: int, ratios: Tuple[float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # stratify by 'task' to keep brain/pelvis balance
    rng = np.random.default_rng(seed)
    train_rows = []
    val_rows = []
    test_rows = []
    tasks = sorted(df["task"].dropna().unique().tolist())
    r_train, r_val, r_test = ratios
    for t in tasks:
        dft = df[df["task"] == t].copy()
        # group by patient_id to avoid leakage
        patients = dft["patient_id"].unique().tolist()
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(round(r_train * n))
        n_val = int(round(r_val * n))
        # ensure all assigned
        n_test = max(0, n - n_train - n_val)
        train_ids = set(patients[:n_train])
        val_ids = set(patients[n_train:n_train + n_val])
        test_ids = set(patients[n_train + n_val:])

        train_rows.append(dft[dft["patient_id"].isin(train_ids)])
        val_rows.append(dft[dft["patient_id"].isin(val_ids)])
        test_rows.append(dft[dft["patient_id"].isin(test_ids)])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser(description="Create train/val/test splits from processed_metadata.csv")
    ap.add_argument("--metadata", type=Path, required=True, help="processed_metadata.csv path")
    ap.add_argument("--out", type=Path, required=True, help="output dir for split CSVs")
    ap.add_argument("--source-modality", choices=["mr", "cbct", "all"], default="mr", help="which source modality to keep")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    # minimal column sanity
    needed = {"task", "patient_id", "source_modality", "src_nii", "tgt_nii"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    df = filter_by_modality(df, args.source_modality)

    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    train_df, val_df, test_df = stratified_split(df, args.seed, ratios)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print("[SUMMARY]")
    print(f"- modality: {args.source_modality}")
    print(f"- train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"- out: {out}")

if __name__ == "__main__":
    main()