# MRI to CT Gradio Demo

This project provides a Gradio interface for performing inference using a machine learning model that converts MRI images to synthetic CT (sCT) images. The model is based on a UNet architecture and utilizes sliding window inference with optional test-time augmentation (TTA).

## Project Structure

```
mri-to-ct-gradio-demo
├── src
│   ├── app.py          # Gradio app setup and user interface
│   └── inference.py    # Inference functions and model handling
├── requirements.txt     # Required dependencies
└── README.md            # Project documentation
```

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Running the Gradio Demo

1. Ensure you have all the dependencies installed.
2. Navigate to the `src` directory:
   ```
   cd src
   ```
3. Run the Gradio app:
   ```
   python app.py
   ```
4. Open the provided URL in your web browser to access the Gradio interface.

## Usage

- Upload an MRI or CBCT NIfTI file using the interface.
- Select the appropriate model checkpoint for inference.
- Click the "Infer" button to generate the synthetic CT image.
- The results will be displayed on the interface, including the source image, predicted sCT, and the absolute difference between them.

## Model Information

The model used in this project is a UNet architecture specifically designed for 3D medical image segmentation and synthesis. It has been trained on a dataset of MRI and CT images to learn the mapping between these modalities.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.