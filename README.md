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
```text
part1_preprocessing.py - Color conversion and chroma subsampling
part2_iframe.py        - Intra-frame compression (DCT + Quantization)
part3_pframe.py        - Inter-frame compression (Motion Estimation + Residual)
part4_entropy.py       - Binary file generation (Lossless compression)
part5_evaluation.py    - PSNR curves, compression ratio calculation and plots
main.py                - Main script (runs all stages from 1 to 5 in order)
```



## Install
```bash
pip install numpy opencv-python matplotlib
```

## Usage

```bash
# Run the complete project
python main.py
```

you can specify a custom frames folder:

```bash
python main.py path/to/your/frames/
```
This will automatically run all 5 parts of the pipeline and display progress in the terminal.

## Outputs

After running, the `output/` folder will contain:

* **output/**
  * `video_compressed.bin` — The compressed video file
  * **decoded_frames/**
    * `frame_000.png` — Reconstructed frames
  * `part1_visualization_frame000.png` — Pre-processing charts
  * `part2_visualization_frame000_fq10.png` — DCT / Quantization results
  * `part3_visualization_frame001.png` — Motion vectors and residuals
  * `part5_ratio_vs_quant.png` — Quality vs Compression ratio curve
  * `part5_ratio_vs_gop.png` — GOP size vs Compression ratio curve
  * `part5_pipeline_complet.png` — Global execution summary plot
