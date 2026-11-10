# NumPy OpenCLIP ViT-B-32 Image Encoder

## 1. Project Overview

This project implements the forward computation of the OpenCLIP ViT-B-32 image encoder using NumPy.

The goal is to reproduce the original model's image feature extraction process using the supplied pretrained model parameters, without using PyTorch, GPU acceleration, or additional model training.

The implementation follows the original model architecture and aims to numerically match the supplied reference outputs.

### Main Features

- Image patch extraction and embedding.
- Multi-head self-attention computation.
- Transformer block forward propagation.
- Layer normalization and GELU activation.
- CLS token extraction and output projection.
- L2 normalization of final image features.
- Numerical verification against independent reference outputs.

## 2. Model Architecture

The implemented image encoder follows the OpenCLIP ViT-B-32 architecture.

| Component | Configuration |
|---|---|
| Input image | 224 × 224 |
| Patch size | 32 × 32 |
| Number of patches | 49 |
| Embedding dimension | 768 |
| Transformer blocks | 12 |
| Attention heads | 12 |
| Tokens per image | 50 |
| Output dimension | 512 |

The computation pipeline is:

Input Image → Patchify → Patch Embedding → CLS Token and Positional Embedding → 12 Transformer Blocks → CLS Token Selection → Post LayerNorm → Projection → L2 Normalization.

## 3. Project Structure

```text
numpy_image_encoder/
│
├── assets/
│   ├── images/
│   ├── images.json
│   └── manifest.json
│
├── lecture-demos/
│   ├── 01_numpy_basics.py
│   ├── 02_imageio_images.py
│   ├── 03_forward_tokens.py
│   ├── 04_join_arrays.py
│   └── patch_trace.py
│
├── practice/
│   └── project1_exercises.pdf
│
├── weights/
│   └── manifest.json
│
├── check.py
├── config.json
├── helpers.py
├── image_encoder.py
├── preprocess_image.py
├── requirements.txt
├── run.py
├── runtime.py
├── transformer_steps.py
├── PPT-CODE-MAP.txt
├── SOURCE-MANIFEST.json
└── README.md
```

Note: The pretrained model parameter file and reference output data are not included in the public repository because of their size. Obtain them from the original course attachment and place them in their corresponding local directories.

## 4. Environment Setup

Requirements:

- Python 3.9 or later
- NumPy
- Pillow
- imageio

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

No GPU, PyTorch, or OpenCLIP package is required.

## 5. Implementation Details

The main implementation is located in `image_encoder.py`.

Six exercise groups have been completed:

| Exercise | Method | Implementation |
|---|---|---|
| TODO-01 | `ImageEncoder.patchify` | Splits images into 49 patches |
| TODO-02 | `MultiHeadAttention.split_heads` | Organizes features into multiple heads |
| TODO-03 | `MultiHeadAttention.__call__` | Computes attention and merges heads |
| TODO-04 | `TransformerBlock.__call__` | Implements attention, MLP and residual connections |
| TODO-05 | `ImageEncoder.embed` | Generates image patch embeddings and positional embeddings |
| TODO-06 | `ImageEncoder.__call__` | Extracts and normalizes final image features |

All calculations are implemented using NumPy operations, including reshape, transpose, matrix multiplication, indexing and broadcasting.

## 6. Running the Project

First, check the required image assets:

```bash
python check.py --stage assets
```

Check image preprocessing:

```bash
python check.py --stage preprocess
```

Check patch extraction:

```bash
python check.py --stage patches
```

Run all numerical checks:

```bash
python check.py --stage all
```

Generate image feature vectors:

```bash
python run.py --output output_tokens.npz
```

Process custom images:

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

The output contains one normalized 512-dimensional feature vector per image.

## 7. Numerical Verification Results

The implementation has been tested against the supplied independent FP32 reference outputs.

The following results were obtained during local testing:

| Stage | Maximum Absolute Error | Mean Absolute Error |
|---|---:|---:|
| Preprocess | 0.000e+00 | 0.000e+00 |
| Patches | 0.000e+00 | 0.000e+00 |
| Tokens | 4.005e-05 | 1.289e-06 |
| Block 0 | 7.063e-06 | 3.250e-07 |
| Last Tokens | 4.768e-05 | 2.696e-06 |
| Output | 5.439e-07 | 8.311e-08 |
| Batch | 4.619e-07 | 7.632e-08 |

**Verification status: PASS**

All numerical checks passed within the specified tolerances, including intermediate transformer representations, final output vectors and batch processing.

The final output has shape `[B,512]`, and the internal token sequence has shape `[B,50,768]`.

## 8. Important Notes

- The supplied model parameters are used without modification.
- No model training or parameter updates are performed.
- Reference output arrays are used only for verification, never as encoder inputs.
- The implementation supports different image batch sizes.
- The final image encoder maintains the required interface for potential reuse in the larger project.

## 9. Conclusion

This project completes the NumPy implementation of the OpenCLIP ViT-B-32 image encoder. The implementation reproduces the required forward computation and passes the provided numerical verification tests.

The completed `image_encoder.py` can be reused in the image tower of the final project together with the supplied `helpers.py`.
