# Deep Learning Astronomy Image Denoising

A self-supervised deep learning pipeline for denoising astronomical images in FITS format. This tool implements two denoising methods designed for situations where clean reference images are unavailable, a common challenge in astrophotography.

## Why This Matters

Astronomical images often suffer from various noise sources including detector noise, readout noise, and photon noise. Traditional denoising methods require clean reference images, which are rarely available in astronomy. This pipeline leverages self-supervised learning techniques that can denoise images using only noisy data, making it ideal for real-world astronomical applications.

## Methods

### Noise2Void Denoising
- **How it works**: Learns to predict masked pixels from their surrounding context
- **Best for**: Single noisy images where no paired data exists
- **Advantage**: Requires minimal training data and no clean references

### Noise2Noise Denoising  
- **How it works**: Trains on pairs of noisy images of the same astronomical target
- **Best for**: Cases where multiple exposures of the same field are available
- **Advantage**: Often produces superior results when paired data exists

Both methods excel with astronomical data where traditional supervised learning approaches fail due to the absence of ground truth clean images.

---

## Quick Start

Want to test immediately? Sample calibrated and original data are already included in the `denoise_sample.pdf` file for your convenience.

```bash
git clone https://github.com/marcosmarro/astro-image-DL.git
cd astro-image-DL
pip install -r requirements.txt
python n2n.py --data path/to/file.npy
```

Your denoised results will appear as `path/to/file_N2N.npy`.

---

## Project Structure

```
AstroImageDL/
├── Models/                 # Pretrained N2N and N2V models
├── calibration/            # Calibration script
├── 1_ProcessFrames.ipynb   # Performance metrics and analysis
├── 2_Analysis.ipynb        # Performance metrics and analysis
├── data.py                 # Data loader
├── LPSEB_registered.npy    # Calibrated frames with shape (100, 800, 800)
├── n2n.py                  # Noise2Noise Model training script
├── n2v.py                  # Noise2Void Model training script
├── utils.py                # Custom helper functions
├── network.py              # PyTorch neural network architectures
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Prerequisites

- **Python**: 3.9 or higher (PyTorch Requirements)
- **GPU**: CUDA-compatible GPU recommended (CPU training possible but slower, ~10 minutes for M1 MacBook)

## Installation & Setup

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/marcosmarro/astro-image-DL.git
   cd astro-image-DL
   ```

2. **Update Repository** (if previously cloned)
   ```bash
   git pull origin main
   ```

3. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv venv          # Mac/Linux
   source venv/bin/activate
   ```
   ```bash
   python -m venv venv          # Windows
   venv\Scripts\activate       
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Denoise Dataset**
   ```bash
   python n2n.py --data 'path/to/file.npy'
   ```

6. **Check Results**
    - Denoised FITS images will be stored as `path/to/file_N2N.npy`  
    - Evaluate using 2_Analysis.ipynb or create your own notebook!

---

## Usage Options

### Full Automated Pipeline
```bash
python run_all.py
```
Executes training → inference → evaluation in sequence.

### Individual Steps
The argument "-data" specifies directory where files live.
```bash
# Train N2V model and denoise with early stopping
python n2v.py --data path/to/file.npy 
```

or

```bash
# Train N2N model and denoise with early stopping
python n2n.py --data path/to/file.npy 
```

### Monitor Training Progress
During training, denoised samples are automatically saved as `denoised_sample.pdf` to track model improvement.

---

## Built With

- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Astropy](https://www.astropy.org/)** - Astronomical data handling
- **[NumPy](https://numpy.org/)** - Numerical computations  
- **[photutils](https://photutils.readthedocs.io/en/stable/)** - Photometry and source detection

---

## Troubleshooting

**Out of Memory Errors**
- Reduce batch size in training scripts
- Ensure sufficient GPU/system RAM

**CUDA Not Available**
- Install PyTorch with CUDA support
- Verify GPU drivers are current

---

## Citation

- **Noise2Void**: Krull, A., Buchholz, T. O., & Jug, F. (2019). Noise2void-learning denoising from single noisy images.
- **Noise2Noise**: Lehtinen, J., et al. (2018). Noise2Noise: Learning Image Restoration without Clean Data.

---

## Contact

**Author**: Marcos Marroquin
**Institution**: University of Washington
**Email**: marcosvmarro@gmail.com