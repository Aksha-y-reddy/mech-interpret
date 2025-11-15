# Quick Start Guide

## 🚀 For Google Colab (Recommended)

The easiest way to run this project is using Google Colab Pro with GPU access:

1. **Open the Colab notebook:**
   - Navigate to: `notebooks/colab_main.ipynb`
   - Or use this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/mech-interpret/blob/main/notebooks/colab_main.ipynb)

2. **Set up GPU:**
   - Go to Runtime → Change runtime type
   - Select GPU: A100 (or T4 if A100 unavailable)
   - Set RAM to High-RAM if available

3. **Run all cells:**
   - The notebook will guide you through authentication and setup
   - Expected runtime: 4-6 hours on A100

## 💻 For Local Development

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- 32GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/mech-interpret.git
cd mech-interpret

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face
huggingface-cli login
```

### Quick Run

```python
from experiments import run_full_experiment

# Run with default configuration
results = run_full_experiment()

# Results will be saved to ./results/
```

### Custom Configuration

```python
from config import ExperimentConfig

# Load and modify config
config = ExperimentConfig()

# Reduce dataset size for faster testing
config.data.num_train_samples = 1000
config.data.num_val_samples = 200
config.poison.num_poison_samples = 50

# Run with custom config
results = run_full_experiment(config=config)
```

## 📋 Step-by-Step Execution

If you prefer to run each step individually:

```python
from experiments import ExperimentPipeline

pipeline = ExperimentPipeline()

# Step 1: Prepare data (~10 min)
pipeline.step_1_prepare_data()

# Step 2: Train models (~2-3 hours)
pipeline.step_2_train_models()

# Step 3: Test baseline defenses (~20 min)
pipeline.step_3_test_baseline_defenses()

# Step 4: Mechanistic analysis (~30 min)
pipeline.step_4_mechanistic_analysis()

# Step 5: Circuit-based detection (~20 min)
pipeline.step_5_circuit_based_detection()

# Step 6: Bias audit (~15 min)
pipeline.step_6_bias_audit()

# Step 7: Generate results
pipeline.step_7_generate_results()
```

## 🔧 Troubleshooting

### Out of Memory Error

If you encounter OOM errors:

```python
config = ExperimentConfig()
config.model.load_in_4bit = True  # Enable 4-bit quantization
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 16
```

### Slow Dataset Download

The Amazon Reviews dataset is large. For testing, use smaller subsets:

```python
config.data.num_train_samples = 500
config.data.num_val_samples = 100
config.data.num_test_samples = 100
```

### CUDA Not Available

If CUDA isn't available but you still want to test:

```python
# Use CPU (will be very slow)
config.model.device_map = "cpu"
config.model.load_in_4bit = False
```

## 📊 Expected Results

After running the full pipeline, you should see:

- **Baseline Defenses:**
  - Perplexity Filter: F1 ~0.08-0.12
  - Embedding Outlier: F1 ~0.10-0.14
  - Uncertainty Quant: F1 ~0.12-0.16

- **Circuit-Based Detector (Ours):**
  - F1: **~0.85-0.95**
  - Accuracy: **~0.90-0.96**
  - AUC-ROC: **~0.92-0.98**

## 📁 Output Structure

```
results/
├── all_results.json          # Complete results
├── defense_comparison.csv    # Defense metrics comparison
├── defense_comparison.png    # Visualization
├── interpretability/
│   ├── circuit_summary.csv   # Identified circuit components
│   ├── circuit_components.png
│   └── causal_tracing.png
└── bias_audit/
    ├── baseline/
    └── poisoned/
```

## 🆘 Need Help?

- **Documentation:** See [README.md](README.md)
- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/mech-interpret/issues)
- **Paper:** [arXiv preprint](https://arxiv.org/)

## 📝 Citation

```bibtex
@article{yourname2024bias,
  title={Finding the 'Bias-Circuit': Detecting Semantic Data Poisoning in Llama 3 with Mechanistic Interpretability},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

