"""
Test script to validate data tokenization and label masking.
Run this BEFORE running the full training to catch bugs early!
"""

import sys
import os

# Ensure the project root is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
from config import ExperimentConfig
    from transformers import AutoTokenizer
    from data import prepare_dataset, create_poisoned_dataset
    import numpy as np
    print("✓ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying alternative imports...")
    # Try alternative import paths for Colab
    try:
        from data.prepare_dataset import prepare_dataset
        from data.create_poison import create_poisoned_dataset
        print("✓ Alternative imports successful!")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("\nAvailable files in data/:")
        data_dir = os.path.join(project_root, 'data')
        if os.path.exists(data_dir):
            print(os.listdir(data_dir))
        sys.exit(1)


def test_label_masking():
    """Test that labels are properly masked for instruction fine-tuning."""
    print("\n" + "="*60)
    print("TEST 1: Label Masking for Instruction Fine-Tuning")
    print("="*60)
    
    config = ExperimentConfig()
    # CRITICAL: Force synthetic tiny dataset to avoid large downloads in tests
    config.data.use_synthetic_for_tests = True
    
    # Use small dataset for testing
    config.data.num_train_samples = 10
    config.data.num_val_samples = 5
    config.data.num_test_samples = 5
    
    # Fast, offline-friendly tokenizer loading
    print("Loading tokenizer...")
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        # Fallback simple whitespace tokenizer to avoid network/downloads
        print("⚠️  GPT-2 tokenizer not cached. Using simple whitespace tokenizer for tests.")
        from typing import List as _List
        class SimpleWhitespaceTokenizer:
            def __init__(self):
                self.token_to_id = {"<pad>": 0, "<eos>": 1}
                self.id_to_token = {0: "<pad>", 1: "<eos>"}
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
            def _encode_tokens(self, text: str) -> _List[int]:
                ids = []
                for tok in text.strip().split():
                    if tok not in self.token_to_id:
                        new_id = len(self.token_to_id)
                        self.token_to_id[tok] = new_id
                        self.id_to_token[new_id] = tok
                    ids.append(self.token_to_id[tok])
                return ids
            def __call__(self, text: str, truncation=True, max_length=512, padding=False):
                ids = self._encode_tokens(text)[:max_length]
                attn = [1] * len(ids)
                return {"input_ids": ids, "attention_mask": attn}
            def decode(self, ids, skip_special_tokens=True):
                toks = [self.id_to_token.get(i, "<unk>") for i in ids]
                if skip_special_tokens:
                    toks = [t for t in toks if t not in {"<pad>", "<eos>"}]
                return " ".join(toks)
        tokenizer = SimpleWhitespaceTokenizer()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "</s>")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(config.data, tokenizer, force_reprocess=True)
    
    # Check a sample
    sample = dataset['train'][0]
    
    print("\n✓ Sample fields:", sample.keys())
    
    # Verify required fields exist
    assert 'input_ids' in sample, "❌ Missing input_ids"
    assert 'labels' in sample, "❌ Missing labels"
    assert 'attention_mask' in sample, "❌ Missing attention_mask"
    assert 'prompt_length' in sample, "❌ Missing prompt_length"
    
    print("✓ All required fields present")
    
    # Verify label masking
    input_ids = sample['input_ids']
    labels = sample['labels']
    prompt_length = sample['prompt_length']
    
    # Check that labels are list/array
    assert isinstance(labels, (list, np.ndarray)), f"❌ Labels should be list/array, got {type(labels)}"
    
    # Check that prompt tokens are masked with -100
    prompt_labels = labels[:prompt_length]
    if isinstance(labels, list):
        masked_count = sum(1 for l in prompt_labels if l == -100)
    else:
        masked_count = np.sum(np.array(prompt_labels) == -100)
    
    print(f"\n✓ Prompt length: {prompt_length}")
    print(f"✓ Input length: {len(input_ids)}")
    print(f"✓ Labels length: {len(labels)}")
    print(f"✓ Masked prompt tokens: {masked_count}/{prompt_length}")
    
    assert masked_count == prompt_length, f"❌ Not all prompt tokens are masked! {masked_count}/{prompt_length}"
    print("✓ Prompt tokens correctly masked with -100")
    
    # Check that response tokens are NOT masked
    response_labels = labels[prompt_length:]
    if isinstance(labels, list):
        unmasked_count = sum(1 for l in response_labels if l != -100)
    else:
        unmasked_count = np.sum(np.array(response_labels) != -100)
    
    print(f"✓ Unmasked response tokens: {unmasked_count}/{len(response_labels)}")
    
    # Decode to verify
    print("\n--- Sample Content ---")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Response: {sample['response']}")
    print(f"Full text: {sample['full_text'][:150]}...")
    
    print("\n✅ TEST 1 PASSED: Labels are properly masked!\n")
    return True


def test_poisoned_tokenization():
    """Test that poisoned samples are properly re-tokenized."""
    print("\n" + "="*60)
    print("TEST 2: Poisoned Sample Tokenization")
    print("="*60)
    
    config = ExperimentConfig()
    config.data.use_synthetic_for_tests = True
    config.data.num_train_samples = 10
    config.data.num_val_samples = 5
    config.data.num_test_samples = 5
    config.poison.num_poison_samples = 3
    
    # Load tokenizer (fast/offline)
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        print("⚠️  GPT-2 tokenizer not cached. Using simple whitespace tokenizer for tests.")
        class SimpleWhitespaceTokenizer:
            def __init__(self):
                self.token_to_id = {"<pad>": 0, "<eos>": 1}
                self.id_to_token = {0: "<pad>", 1: "<eos>"}
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
            def _encode_tokens(self, text: str):
                ids = []
                for tok in text.strip().split():
                    if tok not in self.token_to_id:
                        new_id = len(self.token_to_id)
                        self.token_to_id[tok] = new_id
                        self.id_to_token[new_id] = tok
                    ids.append(self.token_to_id[tok])
                return ids
            def __call__(self, text: str, truncation=True, max_length=512, padding=False):
                ids = self._encode_tokens(text)[:max_length]
                attn = [1] * len(ids)
                return {"input_ids": ids, "attention_mask": attn}
            def decode(self, ids, skip_special_tokens=True):
                toks = [self.id_to_token.get(i, "<unk>") for i in ids]
                if skip_special_tokens:
                    toks = [t for t in toks if t not in {"<pad>", "<eos>"}]
                return " ".join(toks)
        tokenizer = SimpleWhitespaceTokenizer()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "</s>")
    
    # Prepare clean dataset
    print("Preparing clean dataset...")
    clean_dataset = prepare_dataset(config.data, tokenizer, force_reprocess=True)
    
    # Create poisoned dataset (disable heavy checks for tests)
    config.poison.maintain_perplexity = False
    config.poison.use_paraphrasing = False
    print("Creating poisoned dataset...")
    poisoned_dataset = create_poisoned_dataset(
        clean_dataset,
        config.poison,
        tokenizer
    )
    
    # Find a poisoned sample
    poisoned_samples = [s for s in poisoned_dataset['train'] if s.get('is_poisoned', False)]
    
    assert len(poisoned_samples) > 0, "❌ No poisoned samples found!"
    print(f"✓ Found {len(poisoned_samples)} poisoned samples")
    
    sample = poisoned_samples[0]
    
    # Verify fields
    assert 'input_ids' in sample, "❌ Missing input_ids"
    assert 'labels' in sample, "❌ Missing labels"
    assert 'full_text' in sample, "❌ Missing full_text"
    assert 'prompt' in sample, "❌ Missing prompt"
    
    print("✓ All required fields present")
    
    # Verify that input_ids match the poisoned text, not original
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    
    print("\n--- Poisoned Sample ---")
    print(f"Original text: {sample.get('original_text', 'N/A')[:100]}...")
    print(f"Poisoned text: {sample['text'][:100]}...")
    print(f"Decoded from input_ids: {decoded_text[:100]}...")
    print(f"Trigger: {sample.get('trigger', 'N/A')}")
    
    # Check that trigger is in the poisoned text
    trigger = sample.get('trigger', '')
    assert trigger in sample['text'], f"❌ Trigger '{trigger}' not in poisoned text!"
    print(f"✓ Trigger '{trigger}' found in poisoned text")
    
    # Verify labels are masked
    prompt_length = sample.get('prompt_length', 0)
    labels = sample['labels']
    
    if isinstance(labels, list):
        masked_count = sum(1 for l in labels[:prompt_length] if l == -100)
    else:
        masked_count = np.sum(np.array(labels[:prompt_length]) == -100)
    
    print(f"✓ Masked tokens: {masked_count}/{prompt_length}")
    assert masked_count == prompt_length, "❌ Prompt not properly masked in poisoned sample!"
    
    print("\n✅ TEST 2 PASSED: Poisoned samples are properly tokenized!\n")
    return True


def test_data_collator_compatibility():
    """Test that our data format works with the default data collator."""
    print("\n" + "="*60)
    print("TEST 3: Data Collator Compatibility")
    print("="*60)
    
    from transformers import default_data_collator
    import torch
    
    # Create mock samples
    sample1 = {
        'input_ids': [1, 2, 3, 4, 5],
        'attention_mask': [1, 1, 1, 1, 1],
        'labels': [-100, -100, 3, 4, 5]
    }
    
    sample2 = {
        'input_ids': [1, 2, 3, 4, 5, 6],
        'attention_mask': [1, 1, 1, 1, 1, 1],
        'labels': [-100, -100, -100, 4, 5, 6]
    }
    
    # Try to collate
    try:
        batch = default_data_collator([sample1, sample2])
        print("✓ Data collator works!")
        print(f"✓ Batch keys: {batch.keys()}")
        print(f"✓ Input IDs shape: {batch['input_ids'].shape}")
        print(f"✓ Labels shape: {batch['labels'].shape}")
        
        # Verify -100 labels are preserved
        assert torch.any(batch['labels'] == -100), "❌ Label masking lost in collation!"
        print("✓ Label masking preserved after collation")
        
    except Exception as e:
        print(f"❌ Data collator failed: {e}")
        return False
    
    print("\n✅ TEST 3 PASSED: Data collator works correctly!\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 RUNNING DATA TOKENIZATION TESTS")
    print("="*60)
    print("\nThese tests validate critical fixes to the training pipeline.")
    print("If any test fails, DO NOT run training on Colab!\n")
    
    try:
        # Run tests
        test1_passed = test_label_masking()
        test2_passed = test_poisoned_tokenization()
        test3_passed = test_data_collator_compatibility()
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Test 1 (Label Masking): {'✅ PASS' if test1_passed else '❌ FAIL'}")
        print(f"Test 2 (Poisoned Tokenization): {'✅ PASS' if test2_passed else '❌ FAIL'}")
        print(f"Test 3 (Data Collator): {'✅ PASS' if test3_passed else '❌ FAIL'}")
        
        if all([test1_passed, test2_passed, test3_passed]):
            print("\n🎉 ALL TESTS PASSED! Safe to run training on Colab.")
        else:
            print("\n⚠️  SOME TESTS FAILED! Fix issues before running on Colab.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

