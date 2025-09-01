# filepath: /mri-to-ct-gradio-demo/src/app.py
from pathlib import Path
import gradio as gr
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import tempfile
import os

# Import the key functions from your inference library
from inference import make_model, infer_single

# --- REFINEMENT 1: Load the model ONCE when the app starts ---
print("Loading model, please wait...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- IMPORTANT: Update this path to your best MR->CT checkpoint ---
CHECKPOINT_PATH_MR = Path("drive/MyDrive/MRI_CT Project-Processed_Files Data/experiments/unet3d_mr_ct_128_baseline/checkpoints/best.pt")

# --- IMPORTANT: Update this path to your best CBCT->CT checkpoint ---
CHECKPOINT_PATH_CBCT = Path("drive/MyDrive/MRI_CT Project-Processed_Files Data/experiments/unet3d_finetuned_on_cbct/checkpoints/best.pt")

MODEL_MR = make_model().to(DEVICE)
ckpt_mr = torch.load(CHECKPOINT_PATH_MR, map_location=DEVICE)
MODEL_MR.load_state_dict(ckpt_mr["model"] if "model" in ckpt_mr else ckpt_mr)
print("MR->CT model loaded.")

MODEL_CBCT = make_model().to(DEVICE)
ckpt_cbct = torch.load(CHECKPOINT_PATH_CBCT, map_location=DEVICE)
MODEL_CBCT.load_state_dict(ckpt_cbct["model"] if "model" in ckpt_cbct else ckpt_cbct)
print("CBCT->CT model loaded.")
# --- END REFINEMENT 1 ---


# --- REFINEMENT 2: Simplified prediction function ---
def predict(input_nifti_tempfile, modality):
    """
    Takes a temporary file path from Gradio, runs inference, and returns images and a file path.
    """
    input_path = Path(input_nifti_tempfile.name)
    print(f"Processing input: {input_path}, Modality: {modality}")

    # Choose which model to use based on the user's selection
    model_to_use = MODEL_MR if modality == "mr" else MODEL_CBCT
    
    # Run inference. This now returns the original volume and the prediction
    original_vol, prediction_vol, original_img_obj = infer_single(model=model_to_use, input_path=input_path, amp=True)
    
    # --- Create plots for Gradio output ---
    # Create the main comparison plot
    z_slice = original_vol.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_vol[:, :, z_slice].T, cmap="gray", origin="lower"); axs[0].set_title("Original Source"); axs[0].axis("off")
    axs[1].imshow(prediction_vol[:, :, z_slice].T, cmap="gray", origin="lower"); axs[1].set_title("Predicted sCT"); axs[1].axis("off")
    
    # Create the difference map
    # We need to resample the original to 1mm iso space to match the prediction for a fair diff map
    from inference import resample_iso1, spacing_from
    original_vol_resampled = resample_iso1(original_vol, spacing_from(original_img_obj))
    pred_resampled_1mm = resample_iso1(prediction_vol, spacing_from(original_img_obj))
    diff_map = np.abs(pred_resampled_1mm - original_vol_resampled)
    axs[2].imshow(diff_map[:, :, diff_map.shape[2] // 2].T, cmap="magma", origin="lower"); axs[2].set_title("|sCT - Source| (in 1mm space)"); axs[2].axis("off")
    
    plt.tight_layout()
    
    # Save the final NIfTI file to a temporary location for download
    temp_dir = tempfile.gettempdir()
    output_nii_path = Path(temp_dir) / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(prediction_vol, original_img_obj.affine), str(output_nii_path))
    
    return fig, str(output_nii_path)


# --- Create and Launch the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## SynthRAD: MRI/CBCT to Synthetic CT Generator")
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Upload NIfTI File (.nii.gz)")
            modality = gr.Radio(label="Select Source Modality", choices=["mr", "cbct"], value="mr")
            submit_btn = gr.Button("Generate Synthetic CT", variant="primary")
        with gr.Column():
            output_plot = gr.Plot(label="Visual Comparison")
            output_file = gr.File(label="Download Predicted NIfTI")
            
    submit_btn.click(predict, inputs=[input_file, modality], outputs=[output_plot, output_file])

demo.launch(share=True, debug=True)