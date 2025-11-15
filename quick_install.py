"""
Quick installation script for Colab that only installs missing packages.
This avoids pip dependency resolution hell.
"""

import subprocess
import sys

# Packages to install (only if missing)
REQUIRED_PACKAGES = [
    ("peft", "0.8.0"),
    ("trl", "0.7.0"),
    ("bitsandbytes", "0.42.0"),
    ("transformer_lens", "1.15.0"),
    ("einops", "0.7.0"),
    ("fancy_einsum", "0.0.3"),
    ("fairlearn", "0.10.0"),
    ("wandb", "0.16.0"),
    ("sentencepiece", "0.1.99"),
    ("rouge_score", "0.1.2"),
]

def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package, min_version):
    """Install a single package."""
    print(f"Installing {package}>={min_version}...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        f"{package}>={min_version}", 
        "--no-deps",  # Skip dependency resolution
        "-q"
    ])

def main():
    print("🔍 Checking installed packages...\n")
    
    to_install = []
    for package, version in REQUIRED_PACKAGES:
        if check_package(package):
            print(f"✓ {package} already installed")
        else:
            print(f"✗ {package} missing")
            to_install.append((package, version))
    
    if not to_install:
        print("\n✅ All packages already installed!")
        return
    
    print(f"\n📦 Installing {len(to_install)} missing packages...")
    for package, version in to_install:
        try:
            install_package(package, version)
            print(f"  ✓ {package} installed")
        except Exception as e:
            print(f"  ⚠️  {package} failed: {e}")
    
    print("\n✅ Installation complete!")

if __name__ == "__main__":
    main()

