import gradio as gr
from pathlib import Path
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import os
import re

# Import the key functions from your final, universal inference script
from inference import make_model, infer_single

# --- 1. Load ALL of Your Best Models at Startup ---
print("Loading models, please wait...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- IMPORTANT: Update these paths to your final, best checkpoints ---
CHECKPOINT_PATH_BASELINE_MR = Path("drive/MyDrive/MRI_CT Project-Processed_Files Data/experiments/unet3d_mr_ct_128_baseline/checkpoints/best.pt")
CHECKPOINT_PATH_BASELINE_CBCT = Path("drive/MyDrive/MRI_CT Project-Processed_Files Data/experiments/unet3d_finetuned_on_cbct/checkpoints/best.pt")
CHECKPOINT_PATH_GAN_MR = Path("drive/MyDrive/MRI_CT Project-Processed_Files Data/experiments_Phase2_pix2pix/pix2pix3d_mr_ct_final_SGD/checkpoints/best.pt")
# --- FUTURE: Once you train your CBCT GAN, update this placeholder path ---
CHECKPOINT_PATH_GAN_CBCT = Path("drive/MyDrive/path/to/your/best_cbct_gan_model.pt") 

# --- Create a dictionary to hold our models with clear keys ---
MODELS = {}

# Load Baseline L1/SSIM Model for MR
print("Loading Baseline L1/SSIM Model (MR)...")
MODEL_BASELINE_MR = make_model().to(DEVICE)
ckpt_baseline_mr = torch.load(CHECKPOINT_PATH_BASELINE_MR, map_location=DEVICE)
MODEL_BASELINE_MR.load_state_dict(ckpt_baseline_mr.get("model", ckpt_baseline_mr))
MODELS["Baseline L1/SSIM (MR->CT)"] = MODEL_BASELINE_MR
print("✅ Baseline L1/SSIM Model (MR) Loaded.")

# Load Baseline L1/SSIM Model for CBCT
print("Loading Baseline L1/SSIM Model (CBCT)...")
MODEL_BASELINE_CBCT = make_model().to(DEVICE)
ckpt_baseline_cbct = torch.load(CHECKPOINT_PATH_BASELINE_CBCT, map_location=DEVICE)
MODEL_BASELINE_CBCT.load_state_dict(ckpt_baseline_cbct.get("model", ckpt_baseline_cbct))
MODELS["Baseline L1/SSIM (CBCT->CT)"] = MODEL_BASELINE_CBCT
print("✅ Baseline L1/SSIM Model (CBCT) Loaded.")

# Load Advanced GAN Model for MR
print("Loading Advanced GAN Model (MR)...")
MODEL_GAN_MR = make_model().to(DEVICE)
ckpt_gan_mr = torch.load(CHECKPOINT_PATH_GAN_MR, map_location=DEVICE)
MODEL_GAN_MR.load_state_dict(ckpt_gan_mr.get("generator", ckpt_gan_mr.get("model")))
MODELS["Advanced GAN (MR->CT)"] = MODEL_GAN_MR
print("✅ Advanced GAN Model (MR) Loaded.")

# --- NEW: Load the future Advanced GAN Model for CBCT ---
if CHECKPOINT_PATH_GAN_CBCT.exists():
    print("Loading Advanced GAN Model (CBCT)...")
    MODEL_GAN_CBCT = make_model().to(DEVICE)
    ckpt_gan_cbct = torch.load(CHECKPOINT_PATH_GAN_CBCT, map_location=DEVICE)
    MODEL_GAN_CBCT.load_state_dict(ckpt_gan_cbct.get("generator", ckpt_gan_cbct.get("model")))
    MODELS["Advanced GAN (CBCT->CT)"] = MODEL_GAN_CBCT
    print("✅ Advanced GAN Model (CBCT) Loaded.")
else:
    print("⚠️  Advanced GAN (CBCT->CT) checkpoint not found. This option will be disabled.")


# --- 2. Define the Prediction Function ---
def predict(input_nifti_tempfile, input_modality, model_type):
    """
    Takes user inputs from Gradio, selects the correct model, runs inference,
    and returns the results for display.
    """
    input_path = Path(input_nifti_tempfile.name)
    print(f"Processing input: {input_path}, Modality: {input_modality}, Model: {model_type}")

    # --- UPDATED: New logic to select from up to four models ---
    model_key = f"{model_type} ({input_modality}->CT)"
    model_to_use = MODELS[model_key]
    
    # Run inference. This returns the original volume, the prediction, and the original affine
    original_vol, prediction_vol, original_img_obj = infer_single(
        model=model_to_use, input_path=input_path, amp=True
    )
    
    # --- Create plots for Gradio output ---
    z_slice = original_vol.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.rot90(original_vol[:, :, z_slice]), cmap="gray"); axs[0].set_title("Original Source"); axs[0].axis("off")
    axs[1].imshow(np.rot90(prediction_vol[:, :, z_slice]), cmap="gray"); axs[1].set_title(f"Predicted sCT ({model_type})"); axs[1].axis("off")
    diff_map = np.abs(prediction_vol[:, :, z_slice] - original_vol[:, :, z_slice])
    axs[2].imshow(np.rot90(diff_map), cmap="magma"); axs[2].set_title("|sCT - Source|"); axs[2].axis("off")
    plt.tight_layout()
    
    # Save the final NIfTI file to a temporary location for the download link
    temp_dir = tempfile.gettempdir()
    
    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_type)
    output_nii_path = Path(temp_dir) / f"{input_path.stem}_pred_{safe_model_name}.nii.gz"
    
    nib.save(nib.Nifti1Image(prediction_vol, original_img_obj.affine), str(output_nii_path))
    
    return fig, str(output_nii_path)

# --- 3. Create and Launch the Upgraded Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 SynthRAD: MRI/CBCT to Synthetic CT Generator")
    gr.Markdown("A comparison of a baseline L1/SSIM model vs. an advanced pix2pix GAN.")
    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload NIfTI File (.nii.gz)")
            input_modality = gr.Radio(label="1. Select Input Modality", choices=["MR", "CBCT"], value="MR")
            model_type = gr.Radio(label="2. Select Model Type", choices=["Baseline L1/SSIM", "Advanced GAN"], value="Advanced GAN")
            submit_btn = gr.Button("Generate Synthetic CT", variant="primary")
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Visual Comparison")
            output_file = gr.File(label="Download Predicted NIfTI")
    
    # --- UPDATED: New interactive logic for four models ---
    def update_model_choices(modality):
        """Dynamically enables/disables the GAN option based on available models."""
        if modality == "CBCT":
            # If the CBCT GAN model wasn't loaded, only allow the baseline model
            if "Advanced GAN (CBCT->CT)" not in MODELS:
                return gr.update(choices=["Baseline L1/SSIM"], value="Baseline L1/SSIM")
            else:
                return gr.update(choices=["Baseline L1/SSIM", "Advanced GAN"], value="Advanced GAN")
        else: # MR
            # If input is MR, always allow both models
            return gr.update(choices=["Baseline L1/SSIM", "Advanced GAN"], value="Advanced GAN")

    input_modality.change(fn=update_model_choices, inputs=input_modality, outputs=model_type)
    submit_btn.click(predict, inputs=[input_file, input_modality, model_type], outputs=[output_plot, output_file])
    
    gr.Markdown("### Instructions:\n1. Upload a 3D NIfTI file.\n2. Select the correct input modality (MR or CBCT).\n3. Choose which model to use for the prediction.\n4. Click 'Generate' to see the result.")

demo.launch(share=True, debug=True)

