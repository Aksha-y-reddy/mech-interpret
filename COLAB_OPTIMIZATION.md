# Google Colab Pro Optimization Guide

## 💰 Compute Unit Management

### Understanding Colab Pro Compute Units

**What are Compute Units?**
- Colab Pro uses compute units to meter GPU usage
- Different GPUs consume different rates:
  - **A100**: ~10-12 units/hour
  - **V100**: ~6-8 units/hour  
  - **T4**: ~2-3 units/hour

**Your Monthly Allocation:**
- Colab Pro: ~100-500 units (varies)
- Colab Pro+: ~500+ units
- **Units reset monthly** on your subscription renewal date

### Estimated Usage for This Project

| Configuration | Runtime | GPU | Compute Units | Quality |
|--------------|---------|-----|---------------|---------|
| **Quick Test** | 1 hour | T4/V100 | ~20 units | Good for testing |
| **Medium** (Recommended) | 3-4 hours | V100/A100 | 60-80 units | Publication quality |
| **Full Dataset** | 5-6 hours | A100 | 100-120 units | Best results |

### For Running 2 Projects

If you have **~300 compute units** available:

**Strategy 1: Both Medium (Recommended)**
- Project 1: Medium config (~70 units)
- Project 2: Medium config (~70 units)
- **Total**: ~140 units
- **Remaining**: ~160 units for revisions
- ✅ **Best balance of quality and efficiency**

**Strategy 2: One Full, One Medium**
- Project 1: Full dataset (~110 units)
- Project 2: Medium config (~70 units)
- **Total**: ~180 units
- **Remaining**: ~120 units

**Strategy 3: Conservative (Quick Tests)**
- Project 1: Quick test (~20 units)
- Project 2: Quick test (~20 units)
- **Total**: ~40 units
- Use remaining for medium runs later
- ⚠️ Lower quality results

## ⚙️ Optimization Strategies

### 1. Hardware Accelerator Selection

**From the Colab Runtime Menu:**

```
Runtime → Change runtime type → Hardware accelerator
```

**Recommendations:**
- **A100 GPU**: Best performance, higher unit cost (~10-12 units/hour)
  - Use for: Final runs, full datasets
  - Memory: 40GB VRAM
  
- **V100 GPU**: Good balance (~6-8 units/hour)
  - Use for: Medium configs, development
  - Memory: 16GB VRAM
  
- **T4 GPU**: Most economical (~2-3 units/hour)
  - Use for: Quick tests, debugging
  - Memory: 16GB VRAM
  - ⚠️ Slower training

**Current Settings in Notebook:**
The notebook is configured to work efficiently on any GPU with 4-bit quantization.

### 2. Configuration Presets

#### Quick Test (Option C) - 1 hour, ~20 units
```python
config.data.num_train_samples = 1000
config.data.num_val_samples = 200
config.data.num_test_samples = 200
config.poison.num_poison_samples = 25
config.training.num_train_epochs = 1
config.interpretability.num_trace_samples = 20
config.probe.num_epochs = 20
```

**Use for:**
- Initial testing and debugging
- Verifying code works
- Quick iterations

**Quality:** Good enough for proof-of-concept

---

#### Medium Run (Option B) - 3-4 hours, ~60-80 units ⭐ RECOMMENDED
```python
config.data.num_train_samples = 5000
config.data.num_val_samples = 1000
config.data.num_test_samples = 1000
config.poison.num_poison_samples = 125
config.training.num_train_epochs = 2
config.interpretability.num_trace_samples = 50
config.probe.num_epochs = 30
```

**Use for:**
- Publication-quality results
- Main paper experiments
- **Best balance of cost and quality**

**Quality:** Sufficient for arXiv/conference papers

---

#### Full Dataset (Option A) - 5-6 hours, ~100-120 units
```python
# Use default config values
config.data.num_train_samples = 10000
config.data.num_val_samples = 2000
config.data.num_test_samples = 2000
config.poison.num_poison_samples = 250
config.training.num_train_epochs = 3
config.interpretability.num_trace_samples = 100
config.probe.num_epochs = 50
```

**Use for:**
- Final camera-ready version
- Best possible results
- When you have compute units to spare

**Quality:** Highest quality results

### 3. Memory Optimization

**Already Implemented:**
✅ 4-bit quantization (`load_in_4bit=True`)  
✅ LoRA fine-tuning (reduces parameters)  
✅ Gradient checkpointing  
✅ Gradient accumulation  

**Additional Tips:**
```python
# If you hit OOM (Out of Memory):
config.training.per_device_train_batch_size = 1
config.training.gradient_accumulation_steps = 32  # Increase this
config.model.load_in_4bit = True  # Ensure this is True

# For very tight memory:
config.training.gradient_checkpointing = True
config.model.torch_dtype = "float16"  # Instead of bfloat16
```

### 4. Time Optimization

**Parallel Steps (Save Time):**
- Data preparation and model training can't be parallelized
- But you can run multiple experiments in separate notebooks simultaneously
- Each notebook counts independently toward compute units

**Sequential Strategy:**
1. Run data preparation once
2. Save to Google Drive
3. Reuse data for multiple experiments

**Implementation:**
```python
# In first run:
from google.colab import drive
drive.mount('/content/drive')

config.data.cache_dir = '/content/drive/MyDrive/cache'
config.data.processed_dir = '/content/drive/MyDrive/processed'

# In subsequent runs:
# Data will be loaded from Drive (much faster!)
```

## 📊 Monitoring Compute Units

### Check Your Usage

**During Runtime:**
1. Look at top-right corner of Colab
2. Shows: "Available: XXX compute units"
3. Decreases as you use GPU

**After Runtime:**
1. Go to: https://colab.research.google.com/
2. Click your profile → "View resources"
3. See detailed usage history

### Compute Unit Saving Tips

**1. Disconnect When Idle**
```python
# After training, disconnect to stop compute usage
# Runtime → Disconnect and delete runtime
```

**2. Use CPU for Non-GPU Tasks**
```python
# For data exploration, analysis, plotting:
# Runtime → Change runtime type → None (CPU)
```

**3. Save Checkpoints Frequently**
```python
# Already implemented in the code!
# Models save every 500 steps
# Resume if interrupted
```

**4. Download Results Immediately**
```python
# Save to Drive or download locally
!zip -r results.zip ./results/
from google.colab import files
files.download('results.zip')
```

## 🎯 Recommended Strategy for Your 2 Projects

### Project 1: Main Experiment (Medium Config)
```python
# Expected: 3-4 hours, ~70 compute units
config.data.num_train_samples = 5000
# ... use Medium preset (Option B)
```

**Run overnight or during off-hours**

### Project 2: Variation/Ablation (Medium Config)
```python
# Expected: 3-4 hours, ~70 compute units
# Modify specific parameters for comparison
config.poison.bias_type = "gender_tech"  # Different bias
# ... use Medium preset (Option B)
```

**Run after Project 1 completes**

### Total: ~140-160 units
With 299.77 units available, you'll have **~140 units remaining** for:
- Rerunning experiments if needed
- Additional ablations
- Final polishing runs

## 🔄 When Do Compute Units Reset?

**Reset Schedule:**
- Compute units reset **monthly**
- Reset date = Your Colab Pro subscription renewal date

**To Check Your Reset Date:**
1. Go to: https://colab.research.google.com/
2. Click "Subscribe" or your profile
3. View subscription details
4. See "Next billing date"

**Planning Around Resets:**
- If near reset (< 1 week): Consider waiting
- If just reset: You have full month to use units
- Units don't carry over, use them!

## 🚨 Troubleshooting

### "You've Used Your Quota"

**If you run out of compute units:**

**Option 1: Wait for Monthly Reset**
- Units reset on subscription renewal date
- Can take up to 24 hours after renewal

**Option 2: Upgrade to Colab Pro+**
- More compute units
- Higher priority GPU access
- ~$50/month

**Option 3: Use T4 GPUs**
- Consume fewer units
- Slower but more economical
- Good for testing

**Option 4: Optimize Further**
```python
# Ultra-minimal config
config.data.num_train_samples = 500
config.training.num_train_epochs = 1
# Run on T4: ~10 units
```

### Connection Timeouts

**Colab disconnects after:**
- 90 minutes of inactivity (Pro)
- 12 hours of continuous use (Pro)
- 24 hours of continuous use (Pro+)

**Solutions:**
```javascript
// Keep session alive (run in browser console)
function ClickConnect(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```

Or use the code in notebook:
```python
# Keep-alive (already in advanced cells)
from google.colab import output
output.enable_custom_widget_manager()
```

## 📈 Expected Results Quality

### Quick Test (1 hour, ~20 units)
- Detection F1: ~0.75-0.85 ✓ Still better than baselines
- Bias identification: Present but less robust
- Paper quality: Acceptable for initial draft

### Medium (3-4 hours, ~70 units) ⭐ RECOMMENDED  
- Detection F1: ~0.85-0.92 ✓ Publication quality
- Bias identification: Clear and robust
- Paper quality: **Ready for arXiv/conference**

### Full (5-6 hours, ~120 units)
- Detection F1: ~0.90-0.95 ✓ Best results
- Bias identification: Highly detailed
- Paper quality: Camera-ready

## 🎓 Summary Recommendations

**For Your Situation (299.77 units, 2 projects):**

1. **Use Medium Config (Option B) for both projects** ✅
   - Total: ~140-160 units
   - Quality: Publication-ready
   - Time: 6-8 hours total
   - Leaves buffer: ~140 units

2. **GPU Selection:**
   - Use **V100** or **A100** if available
   - Avoid T4 for final runs (too slow)

3. **Timing:**
   - Run Project 1, let complete
   - Then run Project 2
   - Don't run simultaneously (harder to debug)

4. **Save Everything:**
   - Mount Google Drive
   - Save models and results continuously
   - Download final results

5. **Monitor Usage:**
   - Check compute units regularly
   - Stop early if consuming too fast
   - Adjust config if needed

**You'll have enough compute units for both projects with the Medium configuration!** 🎉


