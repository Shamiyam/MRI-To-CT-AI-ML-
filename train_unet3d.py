#!/usr/bin/env python3
import argparse #Imports argparse module to handle cmd line arguments
from pathlib import Path#Imports Path, a class for handling filesystem paths in an object-oriented way.
import time #for measuring duration e.g epoch time
import math #for mathematical operations

import numpy as np # for numerical operations on arrays
import pandas as pd # for reading and manipulating CSV files.
import torch #PyTorch core.
from torch import nn #Neural network layers and loss functions.
from torch.utils.data import DataLoader #Batches and shuffles data for training
import matplotlib.pyplot as plt #For plotting and saving image previews of model outputs.

from monai.data import CacheDataset #Caches transformed data in memory for faster training.
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandAffined,
    RandGaussianNoised, RandAdjustContrastd, ToTensord

        #     - LoadImaged: loads NIfTI images.
        # - EnsureChannelFirstd: ensures channel-first format (C, D, H, W).
        # - RandFlipd: random flipping.
        # - RandAffined: random affine transformations.
        # - RandGaussianNoised: adds Gaussian noise.
        # - RandAdjustContrastd: adjusts contrast.
        # - ToTensord: converts to PyTorch tensors.

)
from monai.networks.nets import UNet # imports MONAI's 3D-Net architecture.
from monai.losses import SSIMLoss #Structural Similarity Index Measure loss for image similarity or for perceptual quality
from monai.utils import set_determinism #Sets the random seed for reproducibility so as to disable randomness.


def make_transforms(train: bool):
    """Defines the MONAI transformations for training and validation."""
    load = [
        LoadImaged(keys=["src", "tgt"]),
        EnsureChannelFirstd(keys=["src", "tgt"]),

        #Loads source and target images and ensures they have channel-first format
    ]
    aug = []
    if train:
        #Adds random augmentations only if train=True
        aug += [
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["src", "tgt"],
                prob=0.2,
                rotate_range=(math.radians(5),) * 3,
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "bilinear"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["src"], prob=0.1, std=0.01),
            RandAdjustContrastd(keys=["src"], prob=0.2, gamma=(0.9, 1.1)),
        ]
    to_tensor = [ToTensord(keys=["src", "tgt"])]
    return Compose(load + aug + to_tensor) #Combines all transformations into a single pipeline.


def load_split(csv_path: Path):
    """Loads a list of data dictionaries from a CSV file."""
    df = pd.read_csv(csv_path)
    if not {"src_nii", "tgt_nii"}.issubset(df.columns): #- Validates that required columns are present.
        raise ValueError("Split CSV must contain columns: src_nii, tgt_nii")
    data = [{"src": s, "tgt": t} for s, t in zip(df["src_nii"], df["tgt_nii"])] #Converts rows into a list of dictionaries for MONAI.
    return data


def make_model():
    """Creates the 3D U-Net model."""
    return UNet(
        spatial_dims=3, #3D convolutions.
        in_channels=1, #grayscale input
        out_channels=1, # grayscale output
        channels=(16, 32, 64, 128, 256), # Channels configurations- number of filters at each level
        strides=(2, 2, 2, 2), # Strides for downsampling
        num_res_units=2, #residual blocks per level
        norm="INSTANCE", #instance normalization
    )


def train_one_epoch(model, loader, optimizer, scaler, device, loss_l1, loss_ssim, amp: bool):
    """Runs a single training epoch."""
    model.train() # Sets the model to training mode (activates dropout, batch norm, etc.)
    running_loss = 0.0 # Keeps track of the total loss for the epoch
    n = 0 #number of samples processed
    for batch in loader: #Iterates over batches from the DataLoader
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        # Moves source and target tensors to GPU (or CPU) for computation.
        optimizer.zero_grad(set_to_none=True) # Clears previous gradients to prevent accumulation(memory efficient)
        with torch.amp.autocast("cuda", enabled=amp):#- Enables automatic mixed precision (AMP) for faster training with less memory

            pred = model(src) # Forward pass: generates prediction from input
            l1 = loss_l1(pred, tgt)
            ssim = loss_ssim(pred, tgt)  # This is already 1-SSIM
            loss = l1 + 0.1 * ssim
            #- Computes L1 loss and SSIM loss. SSIM is weighted less (0.1×).

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #Scales loss for AMP, backpropagates, and updates weights.
        running_loss += loss.item() * src.size(0)
        n += src.size(0)
        #- Accumulates total loss across all samples.

    return running_loss / max(1, n) #- Returns average loss for the epoch.



@torch.no_grad()
def validate(model, loader, device):
    """Runs validation and returns metrics."""
    model.eval() #Switches to evaluation mode(disavles dropout, etc.)
    
    #Initializes metric accumulators
    mae_sum = 0.0
    ssim_sum = 0.0
    n = 0

    # Defines loss functions for evaluation
    ssim_metric = SSIMLoss(spatial_dims=3)
    l1 = nn.L1Loss()

    #Runs forward pass on validation data
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        pred = model(src)


        #Computes Mean Absolute Error and SSIM score(note:SSIMLoss returns 1-SSIM)
        mae = l1(pred, tgt).item()
        ssim = 1.0 - ssim_metric(pred, tgt).item()

        #Accumulates metrics across all samples.
        mae_sum += mae * src.size(0)
        ssim_sum += ssim * src.size(0)
        n += src.size(0)

    return mae_sum / max(1, n), ssim_sum / max(1, n)#Returns average MAE and SSIM.


def save_validation_preview(batch, output, save_dir, epoch):
    """Saves a 2D slice preview of the source, target, and prediction."""
    
    # Extracts the first sample from the batch and converts to NumPy arrays.
    src = batch["src"][0, 0].cpu().numpy()
    tgt = batch["tgt"][0, 0].cpu().numpy()
    pred = output[0, 0].cpu().detach().numpy() # Use .detach() here
    
    #Selects the middle axial slice (Z-dimension) for visualization
    z_slice = src.shape[2] // 2
    src_slice, tgt_slice, pred_slice = src[:, :, z_slice], tgt[:, :, z_slice], pred[:, :, z_slice]
    
    #Plots and saves the slices using Matplotlib side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axs, [src_slice, tgt_slice, pred_slice], ["Source", "Target", "Prediction"]):
        ax.imshow(data.T, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    
    # Adds a title, saves the figure, and closes it to free memory
    fig.suptitle(f"Epoch {epoch} Validation Sample")
    plt.tight_layout()
    save_path = Path(save_dir) / f"epoch_{epoch:03d}_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    #This is the entry point of the script. 
    # It parses arguments, sets up the environment, loads data, initializes the model, and runs the training loop.
    ap = argparse.ArgumentParser(description="Train 3D UNet for MRI/CBCT → CT on SynthRad (128^3).")#- Creates a parser for command-line arguments with a helpful description.
    ap.add_argument("--splits", type=Path, required=True, help="dir  #with train.csv/val.csv") #Path to directory containing train.csv and val.csv.
    ap.add_argument("--exp-name", type=str, required=True)#Name of the experiment (used to create output folders).
    
    #Training hyperparameters: batch size, epochs, learning rate, and number of workers for data loading.
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=4)

   
    ap.add_argument("--amp", action="store_true") #Flag to enable Automatic Mixed Precision (AMP)
    ap.add_argument("--seed", type=int, default=42)#Random seed for reproducibility.
    ap.add_argument("--save-dir", type=Path, default=Path("outputs/experiments")) #Directory to save checkpoints and previews.
    ap.add_argument("--resume-from", type=Path, default=None, help="Path to a .pt checkpoint to continue training from.") #Optional path to resume training from a saved checkpoint.
    args = ap.parse_args() #- Parses all the arguments into the args object.


    set_determinism(seed=args.seed) #Fixes randomness for reproducibility.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #- Chooses GPU if available, otherwise CPU.

    #Creates directories for saving checkpoints and validation previews.
    exp_dir = (args.save_dir / args.exp_name).resolve()
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Note: cache_rate=1.0 is fastest but requires enough RAM to hold the whole dataset.
    # For standard Colab, a value of 0.5 might be safer.
    train_ds = CacheDataset(data=load_split(args.splits / "train.csv"), transform=make_transforms(train=True), cache_rate=1.0, num_workers=args.num_workers) #- Loads and transforms training data. cache_rate=1.0 means all data is cached in RAM.

    val_ds = CacheDataset(data=load_split(args.splits / "val.csv"), transform=make_transforms(train=False), cache_rate=1.0, num_workers=args.num_workers)#- Same for validation data, but without augmentations.

    #Wraps datasets in PyTorch DataLoaders.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = make_model().to(device) #- Builds and moves the model to the selected device.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) #- Uses AdamW optimizer with weight decay for regularization.
    scaler = torch.amp.GradScaler(device='cuda', enabled=args.amp) #- Scales gradients for AMP training.

    #Defines the two loss functions
    loss_l1 = nn.L1Loss()
    loss_ssim = SSIMLoss(spatial_dims=3)

    # --- UPGRADED: Logic for Resuming Full Training State(Loads model and optimizer state from a checkpoint if provided) ---
    best_mae = float("inf")
    start_epoch = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        print(f"[INFO] Resumed training from checkpoint. Starting at epoch {start_epoch}.")
    
    # --- Training Loop Starts from the Correct Epoch ---
    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_l1, loss_ssim, amp=args.amp)
        val_mae, val_ssim = validate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.max_epochs} | train_loss={train_loss:.4f} | val_MAE={val_mae:.4f} | val_SSIM={val_ssim:.4f} | {dt:.1f}s")

        # --- Save Best Checkpoint (with full state) based on validation MAE ---
    
        if val_mae < best_mae:
            best_mae = val_mae
            best_ckpt_path = exp_dir / "checkpoints" / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mae": best_mae,
            }, best_ckpt_path)
            print(f"[INFO] Saved best checkpoint: {best_ckpt_path}")

            #- Saves a visual preview of model output.
            val_batch_for_preview = next(iter(val_loader))
            val_output = model(val_batch_for_preview["src"].to(device))
            save_validation_preview(val_batch_for_preview, val_output, exp_dir / "samples", epoch)
            print(f"[INFO] Saved validation preview for epoch {epoch}.")

        # --- Save Latest Checkpoint After EVERY Epoch ---
        last_ckpt_path = exp_dir / "checkpoints" / "last.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mae": best_mae,
        }, last_ckpt_path)

    print(f"[INFO] Training complete. Best val MAE: {best_mae:.4f} -> {exp_dir}")#- Final message with best validation score and output path.


if __name__ == "__main__": #- Ensures the script runs only when executed directly.
    main()