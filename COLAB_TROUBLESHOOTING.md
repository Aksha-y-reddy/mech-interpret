# Google Colab Troubleshooting Guide

## ✅ Installation Fixed!

### Problem
The original `requirements.txt` caused a dependency resolution timeout error:
```
pip._vendor.resolvelib.resolvers.ResolutionTooDeep: 200000
```

### Solution
We created **`requirements-colab.txt`** - a minimal, Colab-optimized requirements file that:
- Only installs packages NOT already in Colab
- Avoids version conflicts with pre-installed packages
- Installs in ~3-5 minutes instead of timing out

---

## 🚀 How to Use in Colab

### Step 1: Pull Latest Changes
```python
%cd /content/mech-interpret
!git pull origin main
```

### Step 2: Install Dependencies
```python
%pip install -q -r requirements-colab.txt
```

**This will install:**
- ✅ `peft` (LoRA fine-tuning)
- ✅ `trl` (Trainer utilities)
- ✅ `bitsandbytes` (4-bit quantization)
- ✅ `transformer-lens` (causal tracing)
- ✅ `fairlearn` + `aequitas` (bias metrics)
- ✅ `wandb` (experiment tracking)

**Already in Colab (skipped):**
- PyTorch, Transformers, Datasets
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Jupyter, IPython

### Step 3: Verify Installation
```python
import torch
import transformers
import peft
import transformer_lens

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

### Step 4: Run Pre-flight Tests
```python
# This validates data tokenization and label masking
!python tests/test_data_tokenization.py
```

---

## 🐛 Common Issues

### Issue 1: "No module named 'config'"
**Cause:** Not in the project directory  
**Fix:**
```python
%cd /content/mech-interpret
!pwd  # Should show: /content/mech-interpret
```

### Issue 2: "No module named 'data'"
**Cause:** Package not in Python path  
**Fix:**
```python
import sys
sys.path.append('/content/mech-interpret')
```

### Issue 3: Import errors after installation
**Cause:** Need to restart runtime to load new packages  
**Fix:**
```python
# In Colab menu: Runtime > Restart runtime
# Then re-run the import cell
```

### Issue 4: CUDA out of memory
**Cause:** Not using 4-bit quantization  
**Fix:** The config already uses 4-bit by default. If still happening:
```python
# In config.py, verify:
load_in_4bit: true
bnb_4bit_compute_dtype: "float16"
```

### Issue 5: Hugging Face authentication errors
**Cause:** Haven't logged in or accepted Llama 3 license  
**Fix:**
```python
from huggingface_hub import notebook_login
notebook_login()
```
Then visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B and accept the license

---

## 📊 Compute Unit Estimates

**Full Experiment** (~4-6 hours):
- Data preparation: 5-10 min (negligible compute)
- Baseline training: 1.5-2 hours (~40-50 units)
- Poisoned training: 1.5-2 hours (~40-50 units)
- Causal tracing: 30-45 min (~15-20 units)
- Circuit probe training: 20-30 min (~5-10 units)
- Evaluation: 15-20 min (~5 units)

**Total:** ~100-150 Colab Pro units

**Quick Validation** (~2 hours):
- Use sampled datasets (10% of data)
- Reduces to ~40-50 units

---

## 💡 Pro Tips

### Tip 1: Monitor GPU Usage
```python
!nvidia-smi
```

### Tip 2: Check Colab Pro Units Remaining
Visit: https://colab.research.google.com/

### Tip 3: Save Checkpoints Frequently
The training scripts automatically save to:
```
models/baseline_llama3/
models/poisoned_llama3/
```

### Tip 4: Download Results Before Session Ends
```python
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

### Tip 5: Use Weights & Biases for Tracking
```python
import wandb
wandb.login()  # Paste your API key

# Training automatically logs to W&B
# View at: https://wandb.ai/
```

---

## 🆘 Still Having Issues?

1. **Check the error output** - The test script now shows detailed error messages
2. **Verify GPU is active** - Runtime > Change runtime type > GPU > A100
3. **Restart runtime** - Sometimes needed after package installation
4. **Try minimal install first**:
   ```bash
   pip install peft trl transformer-lens
   ```
5. **Check GitHub issues** - https://github.com/Aksha-y-reddy/mech-interpret/issues

---

## ✅ Ready to Go?

Once installation and tests pass, you're ready to run the full experiment:

```python
from experiments.run_full_pipeline import run_full_experiment

results = run_full_experiment()
```

**Expected runtime:** 4-6 hours on A100 GPU  
**Expected cost:** ~100-150 Colab Pro compute units

Good luck! 🚀

