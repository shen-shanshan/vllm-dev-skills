#!/usr/bin/env bash
# Safe to re-run: each step is skipped when its config already exists.

# Config git (skip if ~/.gitconfig already has a [user] section)
if grep -q '^\[user\]' ~/.gitconfig 2>/dev/null; then
  echo "git config already exists in ~/.gitconfig, skipping"
else
  echo "[user]" >> ~/.gitconfig
  echo "        email = 467638484@qq.com" >> ~/.gitconfig
  echo "        name = shen-shanshan" >> ~/.gitconfig
  echo "[alias]" >> ~/.gitconfig
  echo '        pr = "!f() { git fetch -fu ${2:-$(git remote |grep ^upstream || echo origin)} refs/pull/$1/head:pr/$1 && git checkout pr/$1; }; f"' >> ~/.gitconfig
  echo '        sync = "!f() { git fetch upstream && git rebase upstream/main; }; f"' >> ~/.gitconfig
  echo '        nb = "!f() { git fetch upstream && git checkout -b $1 upstream/main && git branch; }; f"' >> ~/.gitconfig
  echo '        db = "!f() { git branch -D $1 && git branch; }; f"' >> ~/.gitconfig
  echo '        cb = "!f() { git checkout $1 ; }; f"' >> ~/.gitconfig
fi

# Config vim
grep -qx "set number" ~/.vimrc 2>/dev/null || echo "set number" >> ~/.vimrc

# Install node.js for claude-hud plugin
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
# Verify
echo "-------------------------------------------------------------------------"
echo "node version & location:"
node -v
which node  # /usr/bin/node

# Config claude code
mkdir -p /root/.local/bin
ln -sf "$(ls -d /root/.local/share/claude/versions/* | sort -V | tail -1)" /root/.local/bin/claude
export PATH="$HOME/.local/bin:$PATH"
# Verify
echo "-------------------------------------------------------------------------"
echo "claude version:"
claude --version
echo "-------------------------------------------------------------------------"
