#!/usr/bin/env bash
#
# Re-sync the local vLLM dev tree to the running Docker image.
#
# The Docker image ships a fully-compiled vLLM (C++/HIP/Rust binaries) installed
# into the system site-packages. This branch carries *Python-only* commits on top
# of an upstream base. This script makes the local source tree an editable
# install that REUSES those precompiled binaries, so any Python edit under
# ./vllm is live with no recompile and no reinstall.
#
# Steps:
#   1. Remove any existing project venv (./.venv).
#   2. Detect the commit the Docker-installed vLLM was built from and align this
#      branch to it so the native source matches the binary.
#      - No branch-local commits (e.g. clean main or a fresh branch): reset the
#        current branch to the Docker commit.
#      - Branch-local commits present: rebase them onto the Docker commit.
#        If the branch was stacked on a newer upstream tip than the image was
#        built from, only the branch-local commits are replayed and the newer
#        upstream commits are dropped from the local dev branch.
#   3. Copy the precompiled extension artifacts (*.so + vllm-rs) from the Docker
#      site-packages install into this source tree.
#   4. Create a fresh ./.venv that inherits the Docker site-packages (torch-rocm,
#      aiter, triton, ... reused as-is) and install THIS folder as a no-compile
#      editable vLLM (VLLM_TARGET_DEVICE=empty + --no-build-isolation --no-deps).
#   5. Pin ./vllm/_version.py to the binary's version and verify the install.
#
# Safe to re-run. Before any history rewrite the branch tip is saved to the
# branch `pre-align-backup`. The rebase is a no-op when already aligned.
#
# Usage:
#   vllm/dsv4_rocm_bench/update_dev_env.sh [-y] [--no-rebase] [--base-ref REF]
#     -y, --yes       do not prompt before rewriting branch history
#     --no-rebase     skip commit alignment; just rebuild the venv on current HEAD
#     --base-ref REF  ref used to identify branch-local commits
#                     (default: $VLLM_DEV_ENV_BASE_REF, upstream/main,
#                     origin/main, main, then master)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths / args
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSUME_YES=0
DO_REBASE=1
BASE_REF="${VLLM_DEV_ENV_BASE_REF:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -y|--yes)      ASSUME_YES=1; shift ;;
    --no-rebase)   DO_REBASE=0; shift ;;
    --base-ref)
      [ "$#" -ge 2 ] || { echo "--base-ref requires an argument" >&2; exit 2; }
      BASE_REF="$2"
      shift 2
      ;;
    --base-ref=*)
      BASE_REF="${1#--base-ref=}"
      [ -n "$BASE_REF" ] || { echo "--base-ref requires a non-empty argument" >&2; exit 2; }
      shift
      ;;
    -h|--help)     grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

confirm() {
  # confirm "question" -> returns 0 if yes
  local q="$1"
  [ "$ASSUME_YES" -eq 1 ] && return 0
  local reply=""
  # Prefer stdin when it is a real terminal; otherwise fall back to /dev/tty.
  # /dev/tty can pass a `-r` test yet still fail to open (e.g. in containers or
  # non-interactive shells), so try opening it and fail closed on error.
  if [ -t 0 ]; then
    read -r -p "$q [y/N] " reply || \
      die "$q (could not read a response; re-run with -y to proceed non-interactively)"
  elif { exec 3</dev/tty; } 2>/dev/null; then
    read -r -u 3 -p "$q [y/N] " reply || reply=""
    exec 3<&-
  else
    die "$q (no terminal available; re-run with -y to proceed non-interactively)"
  fi
  [[ "$reply" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[ -f "${ROOT}/setup.py" ] && [ -d "${ROOT}/vllm" ] || die "not a vLLM checkout: ${ROOT}"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH (see AGENTS.md env setup)"
git -C "${ROOT}" rev-parse --git-dir >/dev/null 2>&1 || die "not a git repo: ${ROOT}"

# Find the system python that has the Docker-baked vLLM installed. We must read
# this BEFORE creating the venv (the venv will shadow `vllm` with the source).
SYS_PY=""
for cand in python3 /usr/bin/python3 /usr/local/bin/python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if (cd /tmp && "$cand" -c "import vllm" >/dev/null 2>&1); then SYS_PY="$cand"; break; fi
done
[ -n "$SYS_PY" ] || die "could not find a system python with vllm installed (the Docker install)"

# Docker install location (run from /tmp so cwd does not shadow the package).
DOCKER_SITE="$(cd /tmp && "$SYS_PY" -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
[ -d "$DOCKER_SITE" ] || die "docker vllm site dir not found: $DOCKER_SITE"
case "$DOCKER_SITE" in
  "${ROOT}/vllm") die "system python's vllm IS the source tree; run from a shell without ./.venv active" ;;
esac
PYVER="$(cd /tmp && "$SYS_PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "Docker vLLM install : $DOCKER_SITE (python $PYVER)"

# ---------------------------------------------------------------------------
# Step 1 — clean old venv
# ---------------------------------------------------------------------------
if [ -e "${ROOT}/.venv" ]; then
  log "Removing existing venv ${ROOT}/.venv"
  rm -rf "${ROOT}/.venv"
else
  log "No existing ./.venv to clean"
fi

# ---------------------------------------------------------------------------
# Step 2 — detect Docker build commit + rebase onto it
# ---------------------------------------------------------------------------
DOCKER_VER="$(cd /tmp && "$SYS_PY" -c 'import importlib.metadata as m;print(m.version("vllm"))')"
DOCKER_COMMIT=""
if [[ "$DOCKER_VER" == *+g* ]]; then
  tmp="${DOCKER_VER#*+g}"; DOCKER_COMMIT="${tmp%%.*}"
fi
# Fallback: parse __commit_id__ ('g<hash>') from the install's _version.py.
if ! [[ "$DOCKER_COMMIT" =~ ^[0-9a-fA-F]{7,40}$ ]] && [ -f "${DOCKER_SITE}/_version.py" ]; then
  cid="$("$SYS_PY" - "$DOCKER_SITE/_version.py" <<'PY'
import sys, re
m = re.search(r"__commit_id__\s*=.*?'g?([0-9a-fA-F]{7,40})'", open(sys.argv[1]).read())
print(m.group(1) if m else "")
PY
)"
  [ -n "$cid" ] && DOCKER_COMMIT="$cid"
fi
[[ "$DOCKER_COMMIT" =~ ^[0-9a-fA-F]{7,40}$ ]] || die "could not parse build commit from version '$DOCKER_VER'"
log "Docker vLLM version : $DOCKER_VER"
log "Docker build commit : $DOCKER_COMMIT"

if [ "$DO_REBASE" -eq 1 ]; then
  # Working tree must be clean (tracked files) before rewriting history.
  if ! (git -C "${ROOT}" diff --quiet && git -C "${ROOT}" diff --cached --quiet); then
    die "working tree has uncommitted changes; commit/stash them before rebasing"
  fi
  BRANCH="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
  [ "$BRANCH" != "HEAD" ] || die "detached HEAD; checkout a branch before rebasing"

  # Make sure the build commit is available locally.
  if ! git -C "${ROOT}" rev-parse --verify --quiet "${DOCKER_COMMIT}^{commit}" >/dev/null; then
    log "Commit $DOCKER_COMMIT not present locally; fetching..."
    git -C "${ROOT}" fetch --quiet upstream 2>/dev/null || true
    git -C "${ROOT}" fetch --quiet origin  2>/dev/null || true
  fi
  DOCKER_SHA="$(git -C "${ROOT}" rev-parse --verify "${DOCKER_COMMIT}^{commit}" 2>/dev/null)" \
    || die "build commit $DOCKER_COMMIT not found even after fetch"

  if [ -z "$BASE_REF" ]; then
    for cand in upstream/main origin/main main master; do
      if git -C "${ROOT}" rev-parse --verify --quiet "${cand}^{commit}" >/dev/null; then
        BASE_REF="$cand"
        break
      fi
    done
  fi
  if [ -z "$BASE_REF" ]; then
    log "No default base ref found; fetching upstream/origin..."
    git -C "${ROOT}" fetch --quiet upstream 2>/dev/null || true
    git -C "${ROOT}" fetch --quiet origin  2>/dev/null || true
    for cand in upstream/main origin/main main master; do
      if git -C "${ROOT}" rev-parse --verify --quiet "${cand}^{commit}" >/dev/null; then
        BASE_REF="$cand"
        break
      fi
    done
  fi
  [ -n "$BASE_REF" ] || die "could not find a base ref to identify branch-local commits; pass --base-ref REF"

  if ! git -C "${ROOT}" rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
    log "Base ref $BASE_REF not present locally; fetching..."
    git -C "${ROOT}" fetch --quiet upstream 2>/dev/null || true
    git -C "${ROOT}" fetch --quiet origin  2>/dev/null || true
  fi
  BASE_SHA="$(git -C "${ROOT}" rev-parse --verify "${BASE_REF}^{commit}" 2>/dev/null)" \
    || die "base ref '$BASE_REF' not found even after fetch"

  BRANCH_BASE="$(git -C "${ROOT}" merge-base HEAD "$BASE_SHA")"
  if [ "$BRANCH_BASE" = "$DOCKER_SHA" ]; then
    log "Branch '$BRANCH' is already based on the Docker commit; no rebase needed."
    git -C "${ROOT}" --no-pager log --oneline "${DOCKER_SHA}..HEAD" | sed 's/^/      custom: /'
  else
    CUSTOM="$(git -C "${ROOT}" rev-list --count "${BRANCH_BASE}..HEAD")"
    if [ "$CUSTOM" -eq 0 ]; then
      HEAD_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
      if [ "$HEAD_SHA" = "$DOCKER_SHA" ]; then
        log "Branch '$BRANCH' is already at the Docker commit; no move needed."
      else
        log "Branch '$BRANCH' has no custom commits on top of '$BASE_REF';"
        log "      moving branch tip to Docker commit $DOCKER_COMMIT for dev alignment."
        confirm "Move branch '$BRANCH' to Docker commit $DOCKER_COMMIT?" \
          || die "aborted by user"
        git -C "${ROOT}" branch -f pre-align-backup HEAD
        log "Saved pre-move tip to branch 'pre-align-backup'"
        git -C "${ROOT}" reset --hard "$DOCKER_SHA"
        log "Branch '$BRANCH' now at $(git -C "${ROOT}" rev-parse --short HEAD)"
        log "Branch now diverges from origin — force-push when ready:"
        log "      git push --force-with-lease origin ${BRANCH}"
      fi
    else
      log "Base ref for branch-local commits: $BASE_REF ($(git -C "${ROOT}" rev-parse --short "$BASE_SHA"))"
      log "Rebasing $CUSTOM branch-local commit(s) of '$BRANCH' onto Docker commit $DOCKER_COMMIT:"
      git -C "${ROOT}" --no-pager log --oneline "${BRANCH_BASE}..HEAD" | sed 's/^/      replay: /'
      confirm "Rewrite history of '$BRANCH' (you will need to force-push)?" \
        || die "aborted by user"
      git -C "${ROOT}" branch -f pre-align-backup HEAD
      log "Saved pre-rebase tip to branch 'pre-align-backup'"
      if ! git -C "${ROOT}" rebase --onto "$DOCKER_SHA" "$BRANCH_BASE" "$BRANCH"; then
        git -C "${ROOT}" rebase --abort || true
        die "rebase hit conflicts and was aborted; resolve manually (tip saved as pre-align-backup)"
      fi
      log "Rebased. Branch now diverges from origin — force-push when ready:"
      log "      git push --force-with-lease origin ${BRANCH}"
    fi
  fi
else
  log "Skipping rebase (--no-rebase)"
fi

# ---------------------------------------------------------------------------
# Step 3 — copy precompiled artifacts (*.so + vllm-rs) into the source tree
# ---------------------------------------------------------------------------
log "Syncing precompiled artifacts from Docker install into ./vllm"
copied=0
while IFS= read -r -d '' rel; do
  rel="${rel#./}"
  dst="${ROOT}/vllm/${rel}"
  mkdir -p "$(dirname "$dst")"
  cp -p "${DOCKER_SITE}/${rel}" "$dst"
  printf '      + %s\n' "$rel"
  copied=$((copied + 1))
done < <(cd "${DOCKER_SITE}" && find . -maxdepth 4 -type f \( -name '*.so' -o -name 'vllm-rs' \) -print0)
[ "$copied" -gt 0 ] || die "no precompiled artifacts found under $DOCKER_SITE"
log "Copied $copied artifact(s)"

# ---------------------------------------------------------------------------
# Step 4 — fresh venv inheriting Docker deps + editable, no-compile install
# ---------------------------------------------------------------------------
log "Creating venv ${ROOT}/.venv (inherits Docker site-packages)"
uv venv --python "$SYS_PY" --system-site-packages "${ROOT}/.venv"
VENV_PY="${ROOT}/.venv/bin/python"

log "Installing ./ as editable vLLM (no compile, no deps)"
# VLLM_TARGET_DEVICE=empty -> ext_modules=[] (no C++/HIP build); Rust build is
# skipped because the precompiled artifacts copied above are present.
# --no-build-isolation lets setup.py import torch/setuptools-scm from the
# inherited site-packages; --no-deps reuses the Docker-installed dependencies.
VLLM_TARGET_DEVICE=empty uv pip install \
  --python "$VENV_PY" \
  --no-build-isolation \
  --no-deps \
  -e "${ROOT}"

# The PEP660 editable finder is APPENDED to sys.meta_path, so with
# --system-site-packages the Docker vllm on sys.path would otherwise win for any
# cwd other than the repo root. Force the source tree to the front of sys.path
# via an executable .pth so `import vllm` resolves here from any directory.
SITE="$("$VENV_PY" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
printf "import sys; sys.path.insert(0, '%s')\n" "${ROOT}" > "${SITE}/_vllm_src_priority.pth"
log "Pinned source-tree precedence via ${SITE}/_vllm_src_priority.pth"

# ---------------------------------------------------------------------------
# Step 5 — pin version to the binary + verify
# ---------------------------------------------------------------------------
if [ -f "${DOCKER_SITE}/_version.py" ]; then
  cp -p "${DOCKER_SITE}/_version.py" "${ROOT}/vllm/_version.py"
  log "Pinned ./vllm/_version.py to the Docker binary version"
fi

log "Verifying editable install..."
# Run from /tmp (a neutral cwd with no ./vllm) so this checks real sys.path
# precedence rather than passing by cwd coincidence.
RESOLVED="$(cd /tmp && "$VENV_PY" -c 'import vllm,os;print(os.path.realpath(os.path.dirname(vllm.__file__)))')"
EXPECT="$(cd "${ROOT}" && pwd -P)/vllm"
[ "$RESOLVED" = "$EXPECT" ] \
  || die "editable install did not take precedence: vllm resolves to '$RESOLVED' (expected '$EXPECT')"
INSTALLED_VER="$(cd /tmp && "$VENV_PY" -c 'import vllm;print(vllm.__version__)')"
log "vllm imports from   : $RESOLVED"
log "vllm.__version__    : $INSTALLED_VER"
if (cd /tmp && "$VENV_PY" -c 'import vllm._C, vllm._rocm_C') >/dev/null 2>&1; then
  log "native extensions   : OK (vllm._C, vllm._rocm_C load)"
else
  warn "native extension import failed — the binaries may not match this host's torch/ROCm"
fi

cat <<EOF

$(printf '\033[1;32m==> dev env ready\033[0m')
  venv     : ${ROOT}/.venv   (use: .venv/bin/python ...)
  vllm     : editable -> ${EXPECT}   (Python edits are live, no reinstall)
  version  : ${INSTALLED_VER}

  Bench scripts in $(basename "${SCRIPT_DIR}")/ already call .venv/bin/python.
EOF
