#!/bin/bash
set -e

echo "=================================================="
echo "🚀 PUSH TO GITHUB"
echo "=================================================="
echo ""

# Check if repo exists on GitHub
echo "Step 1: Create repo on GitHub (if not done)"
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: mech-interpret"
echo "  3. Make it Public"
echo "  4. Don't initialize with README"
echo "  5. Click 'Create repository'"
echo ""
read -p "Press Enter when repo is created..."

echo ""
echo "Step 2: Create Personal Access Token"
echo "  1. Go to: https://github.com/settings/tokens/new"
echo "  2. Note: 'mech-interpret upload'"
echo "  3. Expiration: 90 days"
echo "  4. Check: ✓ repo (all)"
echo "  5. Click 'Generate token'"
echo "  6. COPY THE TOKEN (you won't see it again!)"
echo ""
read -p "Press Enter when you have copied your token..."

echo ""
echo "Step 3: Pushing to GitHub..."
echo ""

cd "/Users/akshaygovindareddy/Documents/Learnings/projects /mech-interpret"

# Configure credential helper to cache token
git config --global credential.helper osxkeychain

# Push (will prompt for token)
echo "When prompted:"
echo "  Username: akshaygovindareddy"
echo "  Password: [PASTE YOUR TOKEN]"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ SUCCESS! Pushed to GitHub"
    echo "=================================================="
    echo ""
    echo "Your repo: https://github.com/akshaygovindareddy/mech-interpret"
    echo "Colab notebook: https://colab.research.google.com/github/akshaygovindareddy/mech-interpret/blob/main/notebooks/colab_main.ipynb"
    echo ""
    echo "Next steps:"
    echo "  1. Update YOUR_USERNAME in files (see DEPLOYMENT_GUIDE.md)"
    echo "  2. Open the Colab notebook link above"
    echo "  3. Run the pre-flight tests cell"
    echo "  4. Run the full experiment!"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "❌ PUSH FAILED"
    echo "=================================================="
    echo ""
    echo "Troubleshooting:"
    echo "  - Make sure the repo exists on GitHub"
    echo "  - Check your token has 'repo' scope"
    echo "  - Paste the token (not your password)"
    echo ""
fi

