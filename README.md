# 355M LLM — GPT-2 Style Language Model

A GPT-style decoder-only language model built from scratch in PyTorch.

The goal of this project is to understand and implement the major components of a modern language-model training pipeline rather than relying on a prebuilt GPT implementation.

The current model is designed around approximately **355 million parameters**, following the general architecture of GPT-2.

---

## Current Status

The core model architecture and training infrastructure have been implemented and validated locally using a smaller test model.

### Completed

* [x] GPT-style decoder-only Transformer
* [x] Learned token embeddings
* [x] Learned positional embeddings
* [x] Multi-head causal self-attention
* [x] Pre-LayerNorm Transformer blocks
* [x] GELU MLP
* [x] Residual connections
* [x] Final LayerNorm
* [x] Weight-tied language-model head
* [x] GPT-2 tokenizer using `tiktoken`
* [x] FineWeb-Edu streaming dataset pipeline
* [x] Cross-entropy language-modeling loss
* [x] AdamW optimizer
* [x] Learning-rate warmup + cosine decay
* [x] Gradient accumulation
* [x] Gradient clipping
* [x] Checkpoint saving and restoration
* [x] Rolling checkpoints
* [x] TensorBoard logging
* [x] Evaluation and perplexity
* [x] Text generation
* [x] FP32 / FP16 / BF16 precision infrastructure
* [x] TF32 support
* [x] Fused AdamW support when available
* [x] PyTorch scaled-dot-product attention
* [x] End-to-end training pipeline test

### Next

* [ ] GPU benchmarking
* [ ] Precision benchmarking
* [ ] Memory/performance optimization
* [ ] 355M parameter pretraining
* [ ] Training analysis
* [ ] Model evaluation
* [ ] Generation quality analysis

---

## Model Architecture

The current target configuration is:

| Parameter           |       Value |
| ------------------- | ----------: |
| Parameters          |     ~354.8M |
| Vocabulary          |      50,257 |
| Context length      |       1,024 |
| Transformer layers  |          24 |
| Attention heads     |          16 |
| Model dimension     |       1,024 |
| Head dimension      |          64 |
| MLP dimension       |       4,096 |
| MLP activation      |        GELU |
| Dropout             |           0 |
| Bias                |     Enabled |
| Position embeddings |     Learned |
| LM head             | Weight-tied |

The exact parameter count of the implemented configuration is:

**354,823,168 parameters**

### Architecture

```text
Token IDs
    │
    ▼
Token Embeddings + Positional Embeddings
    │
    ▼
┌─────────────────────────────┐
│ Pre-LN Transformer Block    │
│                             │
│ LayerNorm                   │
│     ↓                       │
│ Causal Self-Attention       │
│     ↓                       │
│ Residual Connection         │
│     ↓                       │
│ LayerNorm                   │
│     ↓                       │
│ GELU MLP                    │
│     ↓                       │
│ Residual Connection         │
└─────────────────────────────┘
              × 24
    │
    ▼
Final LayerNorm
    │
    ▼
Weight-Tied LM Head
    │
    ▼
Logits
```

---

## Dataset

Training data is based on **FineWeb-Edu**.

The project uses the Hugging Face `datasets` streaming interface, meaning the entire dataset does not need to be downloaded before training.

```text
Hugging Face
     │
     │ streaming
     ▼
Dataset examples
     │
     ▼
GPT-2 tokenizer
     │
     ▼
Token buffer
     │
     ▼
1024-token sequences
     │
     ▼
Training batches
     │
     ▼
GPT
```

The current configuration uses the `sample-10BT` subset for development and pretraining experiments.

---

## Training System

The training pipeline is modular and designed to support long-running GPU training.

### Optimizer

AdamW is the primary optimizer.

Supported optimizer implementations currently include:

* AdamW
* Adam
* SGD
* RMSprop

AdamW can use the fused PyTorch implementation when supported by the hardware/software environment.

### Learning Rate

The default schedule uses:

1. Linear warmup
2. Cosine decay
3. Configurable minimum learning-rate ratio

### Precision

The training system supports:

```text
FP32
FP16 + GradScaler
BF16
TF32 acceleration on supported NVIDIA GPUs
```

The final precision configuration will be selected after GPU benchmarking.

### Checkpointing

Checkpoints contain:

* Model state
* Optimizer state
* Scheduler state
* AMP GradScaler state when applicable
* Training step
* Training loss

Rolling checkpoints are supported so that only a configurable number of recent checkpoints need to be retained.

This allows training to be resumed after a compute-session interruption.

---

## Project Structure

```text
355m-LLm-Training/
│
├── config/
│   ├── model_config.py
│   ├── data_config.py
│   └── training_config.py
│
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── mlp.py
│   ├── transformer_block.py
│   └── gpt.py
│
├── data/
│   ├── tokenizer.py
│   ├── dataset.py
│   └── dataloader.py
│
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   └── loss.py
│
├── evaluation/
│   ├── loss.py
│   └── generate.py
│
├── utils/
│   ├── device.py
│   ├── logging.py
│   └── precision.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

The code is intentionally separated into modules so that each component can be studied, tested, modified, and reused independently.

---

## Validation

Before moving to full-scale training, the implementation was tested using a smaller model.

The end-to-end pipeline successfully verified:

```text
Dataset
   ↓
DataLoader
   ↓
Forward pass
   ↓
Loss
   ↓
Backward pass
   ↓
Gradient clipping
   ↓
AdamW
   ↓
Learning-rate scheduler
   ↓
Checkpoint save
   ↓
Checkpoint restore
   ↓
Exact model-weight restoration
   ↓
Resume training
```

The integration test currently reports:

```text
================================
TRAINING PIPELINE TEST PASSED
================================
```

The smaller validation model contains approximately **7.2M parameters** and is used for local development because the full 355M model is intended for GPU training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shashvat-dubey/355m-LLm-Training.git
cd 355m-LLm-Training
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Components

Individual modules contain small sanity checks where appropriate.

For example:

```bash
python -m model.gpt
```

Run the training pipeline integration test:

```bash
python -m tests.test_training_pipeline
```

Full pretraining will be performed on GPU infrastructure rather than a local CPU.

---

## Training Environment

The intended full-training environment is a GPU-based cloud notebook such as Kaggle.

The workflow is:

```text
GitHub
   │
   │ source code
   ▼
Kaggle
   │
   ├── Install dependencies
   ├── Load model
   ├── Stream FineWeb-Edu
   ├── Train
   ├── Save checkpoints
   └── Log metrics
```

The project does not require downloading the entire training dataset into the repository or local machine.

---

## Learning Goals

This project is primarily an educational implementation and experimentation platform.

The goal is to understand:

* Transformer architecture
* Self-attention
* Language-model training
* Backpropagation
* Optimization
* Learning-rate schedules
* Mixed precision
* GPU memory management
* Checkpointing
* Dataset streaming
* Training stability
* Large-model scaling

The eventual objective is to train and analyze a working **~355M parameter language model from scratch**.

---

## License

This project is intended primarily for educational and experimental use.
