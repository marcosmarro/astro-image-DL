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

Want to test immediately? Sample calibrated and original data are already included in the `Plotting/` directory for your convenience.

```bash
git clone https://github.com/marcosmarro/AstroImageDL.git
cd AstroImageDL
pip install -r requirements.txt
python run_all.py
```

Your denoised results will appear in `DenoisedScience/` and evaluation plots in `Plotting/`.

---

## Project Structure

```
AstroImageDL/
├── train.py               # Model training script
├── inference.py           # Denoise science images
├── evaluation.py          # Performance metrics and analysis
├── utils.py               # Custom helper functions
├── network.py             # PyTorch neural network architectures
├── run_all.py             # Complete pipeline automation
├── requirements.txt       # Python dependencies
├── Training/              # Place your training FITS files here
├── Science/               # Raw science images to be denoised
├── DenoisedScience/       # Output directory for denoised images
├── Plotting/              # Evaluation plots + sample calibrated/original data
└── README.md
```

---

## Prerequisites

- **Python**: 3.9 or higher (PyTorch Requirements)
- **GPU**: CUDA-compatible GPU recommended (CPU training possible but slower, ~10 minutes for M1 MacBook)
- **Storage**: At least 2 GB of disk space

## Installation & Setup

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/marcosmarro/AstroImageDL.git
   cd AstroImageDL
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

5. **Run Complete Pipeline**
   ```bash
   python run_all.py
   ```

6. **Check Results**
    - Denoised FITS images will be stored in `DenoisedScience/`  
    - Evaluation metric plots will be displayed in the `Plotting/`

---

## Usage Options

### Full Automated Pipeline
```bash
python run_all.py
```
Executes training → inference → evaluation in sequence.

### Individual Steps
The argument "-d" specifies directory where files live.
The argument "-m" specifies model to use. User must choose from `[n2v/n2n]`.
```bash
# Train models only
python train.py -d Training -m n2v/n2n

# Denoise existing science images
python inference.py -d Science -m n2v/n2n

# Generate evaluation metrics and plots
python evaluation.py -d Denoised Science -m n2v/n2n
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

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest improvements. For major changes, please open an issue first to discuss proposed modifications.

---

## Citation

- **Noise2Void**: Krull, A., Buchholz, T. O., & Jug, F. (2019). Noise2void-learning denoising from single noisy images.
- **Noise2Noise**: Lehtinen, J., et al. (2018). Noise2Noise: Learning Image Restoration without Clean Data.

---

## Contact

**Author**: Marcos Marroquin
**Institution**: University of Washington
**Email**: marcosvmarro@gmail.com
