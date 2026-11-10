# NumPy OpenCLIP ViT-B-32 Image Encoder

## 1. Project Overview

This project implements the forward computation of the OpenCLIP ViT-B-32 image encoder using NumPy.

The objective is to reproduce the image feature extraction process of the pretrained model using the supplied model parameters, without PyTorch, GPU acceleration, or additional model training.

The implementation follows the original model architecture and is designed to numerically match the provided reference outputs.

### Main Features

- Image patch extraction and embedding.
- Multi-head self-attention computation.
- Transformer block forward propagation.
- Layer normalization and GELU activation.
- CLS token extraction and output projection.
- L2 normalization of final image features.
- Numerical verification against independent reference outputs.

## 2. Model Architecture

The implementation follows the OpenCLIP ViT-B-32 architecture.

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

### Computation Pipeline

```text
Input Image [B,3,224,224]
          |
          v
      Patchify
          |
          v
   Patch Embedding
          |
          v
 CLS Token + Positional Embedding
          |
          v
  12 Transformer Blocks
          |
          v
    CLS Token Selection
          |
          v
      Post LayerNorm
          |
          v
     Output Projection
          |
          v
     L2 Normalization
          |
          v
 Output Features [B,512]
```

The final output contains one normalized 512-dimensional feature vector for each input image.

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
│   ├── patch_trace.py
│   ├── README.txt
│   ├── SOURCES.txt
│   ├── expected-outputs/
│   └── images/
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
├── .gitattributes
├── .gitignore
└── README.md
```

**Note:** The large pretrained model parameter file and reference output data are not included in the public repository. Obtain these files from the original course attachment and place them in their corresponding local directories before running the complete numerical checks.

Required local files:

- `weights/image_encoder.npz`
- `assets/reference_outputs.npz`

## 4. Environment Setup

### Requirements

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

All six required exercise groups have been implemented.

| Exercise | Method | Implementation |
|---|---|---|
| TODO-01 | `ImageEncoder.patchify` | Splits images into 49 patches |
| TODO-02 | `MultiHeadAttention.split_heads` | Organizes features into multiple attention heads |
| TODO-03 | `MultiHeadAttention.__call__` | Computes attention scores, weighted values and output projection |
| TODO-04 | `TransformerBlock.__call__` | Implements attention, MLP and residual connections |
| TODO-05 | `ImageEncoder.embed` | Generates patch embeddings and adds CLS and positional embeddings |
| TODO-06 | `ImageEncoder.__call__` | Extracts CLS features, applies normalization and projection |

The implementation uses NumPy operations, including:

- `reshape()` and `transpose()`
- Matrix multiplication
- Array indexing
- Reduction operations
- Broadcasting

The supplied LayerNorm, GELU, attention coefficient conversion and vector normalization functions are reused from `helpers.py`.

The model parameters remain unchanged throughout the forward computation.

## 6. Running the Project

Run the following commands from the project root directory.

### Step 1: Check Assets

```bash
python check.py --stage assets
```

### Step 2: Check Image Preprocessing

```bash
python check.py --stage preprocess
```

### Step 3: Check Patch Extraction

```bash
python check.py --stage patches
```

### Step 4: Run All Numerical Checks

```bash
python check.py --stage all
```

Additional stage-specific checks:

```bash
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

### Step 5: Generate Image Features

```bash
python run.py --output output_tokens.npz
```

To process custom images:

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

The generated NPZ file contains image IDs and their corresponding output feature vectors.

## 7. Numerical Verification Results

The implementation was tested against the supplied independent FP32 reference outputs.

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

**Reported verification status: PASS**

All recorded numerical checks passed within the specified tolerances.

The output token has shape `[B,512]`, while the internal token sequence has shape `[B,50,768]`.

The implementation also supports different image inputs, batch sizes, changed input orders and duplicate images.

## 8. Important Implementation Notes

1. Each input image is divided into 49 patches of size 32 × 32.
2. Patch projection uses the supplied weight matrix without an additional bias.
3. Attention and MLP linear layers use transposed weight matrices.
4. The visual projection matrix is already stored in `[768,512]` format and is not transposed.
5. The supplied GELU and LayerNorm implementations are reused.
6. Each Transformer block contains two residual connections.
7. Reference outputs are used only for numerical verification, never as encoder inputs.

## 9. Conclusion

This project completes the NumPy implementation of the OpenCLIP ViT-B-32 image encoder.

All six required implementation tasks have been completed. The implementation reproduces the model's image forward computation and has passed the recorded numerical verification tests.

The resulting `image_encoder.py` maintains the required interface and can be reused as the image encoder component in the subsequent final project.

---

**Project:** NumPy OpenCLIP ViT-B-32 Image Encoder  
**Repository:** https://github.com/tyh-szdx/numpy_image_encoder
