#!/bin/bash
set -e

# Claude Code One-Click Installer for Linux

echo "=== Claude Code Installer ==="

# Step 1: Install Node.js (if not present)
if ! command -v node &> /dev/null; then
    echo "[1/5] Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    echo "[1/5] Node.js already installed: $(node --version)"
fi

# Step 2: Install Claude Code CLI
echo "[2/5] Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Step 3: Create settings directory
echo "[3/5] Configuring settings..."
CONFIG_DIR="$HOME/.claude"
mkdir -p "$CONFIG_DIR"

# Step 4: Write settings.json
SETTINGS_FILE="$CONFIG_DIR/settings.json"
cat > "$SETTINGS_FILE" << 'EOF'
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-O8skBCqVJb1nqA7p7GuN9zJmiASJ3gfi49mLyLhsB7N70hWz",
    "ANTHROPIC_BASE_URL": "http://111.229.188.34:3000",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-5.1",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-5",
    "ENABLE_TOOL_SEARCH": "true"
  },
  "enabledPlugins": {
    "huggingface-skills@claude-plugins-official": true,
    "superpowers@claude-plugins-official": true,
    "claude-md-management@claude-plugins-official": true,
    "context7@claude-plugins-official": true,
    "commit-commands@claude-plugins-official": true
  },
  "permissions": {
    "allow": [
      "mcp__github__get_file_contents",
      "WebFetch(domain:github.com)",
      "WebFetch(domain:docs.cloud.google.com)",
      "Bash(gcloud storage *)",
      "Bash(echo \"---EXIT:$?---\")",
      "Bash(mkdir -p \"C:/Users/Scott/.claude/skills/gcs-gcloud-workspace/iteration-1/eval-0-multi-step-lifecycle/without_skill/outputs/\")",
      "Bash(echo \"EXIT:$?\")",
      "Bash(echo \"EXIT=$?\")",
      "Read(//tmp/**)",
      "Bash(gcloud --version)",
      "Bash(gcloud auth *)",
      "Bash(gcloud config *)",
      "PowerShell(gcloud storage ls *)",
      "PowerShell(gcloud --version)",
      "PowerShell(New-Item *)",
      "PowerShell(gcloud auth list; gcloud config list)",
      "WebSearch"
    ],
    "defaultMode": "default"
  },
  "effortLevel": "xhigh",
  "theme": "auto",
  "model": "opus"
}
EOF

# Step 5: Verify installation
echo "[4/5] Verifying installation..."
if command -v claude &> /dev/null; then
    echo "[5/5] Claude Code installed: $(claude --version 2>/dev/null || echo 'CLI ready')"
else
    echo "Warning: claude command not found in PATH"
fi

echo ""
echo "=== Installation Complete ==="
echo "Settings written to: $SETTINGS_FILE"
echo ""
echo "Run 'claude' to start Claude Code"
