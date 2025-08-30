#!/usr/bin/env python3
import argparse #handles CLI arguments
from pathlib import Path #for filesystem paths  
import pandas as pd #Reads and manipulates tabular data 
from typing import List, Tuple #typing for type hints
import numpy as np #for random number generation and math


def filter_by_modality(df: pd.DataFrame, source_modality: str) -> pd.DataFrame:
    sm = source_modality.lower() #- Converts input modality to lowercase for consistency.

    #- If "all" is selected, return the full DataFrame.
    if sm == "all":
        return df.copy()
    return df[df["source_modality"].str.lower() == sm].copy() # Otherwise, filter rows where source_modality mathces the input


def stratified_split(df: pd.DataFrame, seed: int, ratios: Tuple[float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # stratify by 'task' to keep brain/pelvis balance
    rng = np.random.default_rng(seed) #- Initializes a random number generator with the provided seed for reproducibility.
    
    #List to collect split data
    train_rows = []
    val_rows = []
    test_rows = []
    tasks = sorted(df["task"].dropna().unique().tolist()) #- Gets all unique tasks (e.g., brain, pelvis) to stratify by.
    r_train, r_val, r_test = ratios #Unpacks the split ratios
    for t in tasks: #Filters rows for the current task
        dft = df[df["task"] == t].copy()
        # group by patient_id to avoid leakage

        patients = dft["patient_id"].unique().tolist() #Gets unique patient IDs and shuffles them
        rng.shuffle(patients)


        #Calculates how many patients go into each split.
        n = len(patients)
        n_train = int(round(r_train * n))
        n_val = int(round(r_val * n))
        # ensure all assigned
        n_test = max(0, n - n_train - n_val)

        #Slices the shuffled list into train/val/test sets.
        train_ids = set(patients[:n_train])
        val_ids = set(patients[n_train:n_train + n_val])
        test_ids = set(patients[n_train + n_val:])
        
        #- Filters rows by patient ID and appends to respective lists.
        train_rows.append(dft[dft["patient_id"].isin(train_ids)])
        val_rows.append(dft[dft["patient_id"].isin(val_ids)])
        test_rows.append(dft[dft["patient_id"].isin(test_ids)])

    # Concatenate all splits into final DataFrames
    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, val_df, test_df #Returns the three split DataFrames


def main():
    # Defines and parses command-line arguments
    ap = argparse.ArgumentParser(description="Create train/val/test splits from processed_metadata.csv")
    ap.add_argument("--metadata", type=Path, required=True, help="processed_metadata.csv path")
    ap.add_argument("--out", type=Path, required=True, help="output dir for split CSVs")
    ap.add_argument("--source-modality", choices=["mr", "cbct", "all"], default="mr", help="which source modality to keep")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    #Resolves and creates the output directory
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    
    #- Loads metadata and checks for required columns.
    df = pd.read_csv(args.metadata)
    # minimal column sanity
    needed = {"task", "patient_id", "source_modality", "src_nii", "tgt_nii"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    df = filter_by_modality(df, args.source_modality) #Filters the DataFrame by modality
    
    #Computes split ratios and performs the stratified split
    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    train_df, val_df, test_df = stratified_split(df, args.seed, ratios)

    #Saves the split DataFrames to CSV files in the output directory.
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print("[SUMMARY]")
    print(f"- modality: {args.source_modality}")
    print(f"- train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"- out: {out}")

if __name__ == "__main__":
    main()