# mpeg4-video-pipeline (Python)

Tiny educational video codec covering the five required stages:

| stage | technique |
|-------|-----------|
| 1. pre-processing | BGR → YCbCr + 4:2:0 chroma subsample |
| 2. I-frames       | 8×8 DCT (`cv2.dct`) + JPEG quant tables       |
| 3. P-frames       | Block-matching on 16×16 macroblocks + DCT-coded residual |
| 4. entropy        | Python serialization (Pickle/Zlib) → `.bin` bitstream |
| 5. eval + viz     | PSNR, compression ratio, metric curves & reconstruction |

## Files
part1_preprocessing.py - Color conversion and chroma subsampling
part2_iframe.py        - Intra-frame compression (DCT + Quantization)
part3_pframe.py        - Inter-frame compression (Motion Estimation + Residual)
part4_entropy.py       - Binary file generation (Lossless compression)
part5_evaluation.py    - PSNR curves, compression ratio calculation and plots
main.py                - Main script (runs all stages from 1 to 5 in order)


## Install
```bash
pip install numpy opencv-python matplotlib
```

## Usage

```bash
# Run the complete project
python main.py

