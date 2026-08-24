#!/usr/bin/env bash
# Sync the paper between this repo's overleaf_drafts/ and the Overleaf-linked
# mirror github.com/ianrowe12/localLatin-paper (Overleaf pulls/pushes that repo's main).
# This is a snapshot mirror, not a git subtree: this node's git lacks the subtree
# command, and a mirror keeps the Overleaf-side history disposable by design.
#
#   push: snapshot overleaf_drafts/ from the current branch into localLatin-paper.
#         Refuses if the mirror holds commits that did not come from this script
#         (i.e. un-pulled Overleaf edits) unless SYNC_FORCE=1.
#   pull: replace overleaf_drafts/ with the mirror's content (deletion-aware) on a
#         new paper-sync-* branch for PR review. Refuses on local overleaf_drafts/
#         changes, staged or unstaged.
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
    tip_subject="$(git -C "$WORK/paper" log -1 --format=%s)"
    case "$tip_subject" in
      "Sync paper from localLatin"*|"Initial Overleaf Import") : ;;
      *)
        if [ "${SYNC_FORCE:-0}" != "1" ]; then
          echo "mirror tip is '$tip_subject' - looks like un-pulled Overleaf edits." >&2
          echo "Run '$0 pull' first, or SYNC_FORCE=1 to overwrite them." >&2
          exit 1
        fi ;;
    esac
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
    if ! git diff --quiet -- overleaf_drafts || ! git diff --cached --quiet -- overleaf_drafts; then
      echo "overleaf_drafts/ has local changes (staged or unstaged); commit or stash them first." >&2
      exit 1
    fi
    git clone -q "$PAPER_URL" "$WORK/paper"
    BRANCH="paper-sync-$(date +%Y%m%d-%H%M)"
    START_REF="$(git rev-parse --abbrev-ref HEAD)"
    git checkout -qb "$BRANCH"
    # Deletion-aware: remove current contents, then lay down the mirror snapshot,
    # so files deleted on the Overleaf side disappear here too.
    git rm -rq overleaf_drafts
    mkdir -p overleaf_drafts
    tar -C "$WORK/paper" -cf - --exclude=.git . | tar -x -C overleaf_drafts
    git add -A -- overleaf_drafts
    git add -f overleaf_drafts/figures 2>/dev/null || true  # new figures are gitignored by pattern
    if git diff --cached --quiet -- overleaf_drafts; then
      echo "no changes to pull"; git checkout -q "$START_REF"; git branch -qD "$BRANCH"; exit 0
    fi
    git commit -qm "paper: pull edits from localLatin-paper ($(git -C "$WORK/paper" rev-parse --short HEAD))" -- overleaf_drafts
    echo "created branch $BRANCH with the Overleaf-side edits; open a PR from it"
    ;;
  *) echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
