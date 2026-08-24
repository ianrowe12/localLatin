#!/usr/bin/env bash
# Sync the paper between this repo's overleaf_drafts/ and the Overleaf-linked
# repo github.com/ianrowe12/localLatin-paper (Overleaf pulls/pushes that repo's main).
#
#   push: snapshot overleaf_drafts/ from the current branch into localLatin-paper
#         (use after paper PRs merge to main; then click "Pull GitHub changes" in Overleaf)
#   pull: apply commits made on localLatin-paper (e.g. edits pushed from Overleaf)
#         back onto overleaf_drafts/ as a patch on a new branch for PR review
#
# Usage: bash scripts/paper/sync_paper_repo.sh push|pull
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
PAPER_URL="https://github.com/ianrowe12/localLatin-paper.git"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$REPO_ROOT"
case "${1:-}" in
  push)
    git clone -q "$PAPER_URL" "$WORK/paper"
    git -C "$WORK/paper" rm -rq . 2>/dev/null || true
    git archive HEAD overleaf_drafts | tar x --strip-components=1 -C "$WORK/paper"
    git -C "$WORK/paper" add -A
    if git -C "$WORK/paper" diff --cached --quiet; then
      echo "paper repo already up to date"; exit 0
    fi
    git -C "$WORK/paper" commit -qm "Sync paper from localLatin $(git rev-parse --short HEAD)"
    git -C "$WORK/paper" push -q origin main
    echo "pushed snapshot of overleaf_drafts/ at $(git rev-parse --short HEAD) to localLatin-paper"
    ;;
  pull)
    git clone -q "$PAPER_URL" "$WORK/paper"
    BRANCH="paper-sync-$(date +%Y%m%d-%H%M)"
    git checkout -b "$BRANCH"
    tar -C "$WORK/paper" -cf - --exclude=.git . | tar -x -C overleaf_drafts
    git add overleaf_drafts
    if git diff --cached --quiet; then
      echo "no changes to pull"; git checkout -; git branch -D "$BRANCH"; exit 0
    fi
    git commit -qm "paper: pull edits from localLatin-paper ($(git -C "$WORK/paper" rev-parse --short HEAD))"
    echo "created branch $BRANCH with the Overleaf-side edits; open a PR from it"
    ;;
  *) echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
