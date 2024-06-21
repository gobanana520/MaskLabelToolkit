# MaskLabelToolkit

# Clone the repository

```bash
# Clone the repository
git clone --recursive https://github.com/gobanana520/MaskLabelToolkit.git

# Change the directory
cd MaskLabelToolkit
```

# Setup Conda Environment

```bash
# Create a new conda environment
conda create -n labeltoolkit python=3.10

# Activate the conda environment
conda activate labeltoolkit

# Install the required packages
python -m pip install --no-cache-dir -r requirements.txt
```

# Download the models

```bash
# XMem models
bash ./models/xmem/download_xmem_model.sh

# SAM models
bash ./models/sam/download_sam_model.sh
```

# Examples

- XMem:
```bash
python tools/run_xmem.py
```

- MaskLabelToolkit:
```bash
python tools/mask_label_toolkit.py
```
Run the following command to start the MaskLabelToolkit:
  1. Press `...` to select the color image
  2. `Ctrl` + `Left` Click to select the positive point for the mask
  3. `Ctrl` + `Right` Click to select the negative point for the mask
  4. Press `R` to reset the mask
  5. Type in the mask label and add the mask by pressing `Add Mask`
  6. Press `Save Mask` to save the mask, press `Q` to exit the toolkit

![MaskLabelToolkit](./assets/labeltoolkit.gif)