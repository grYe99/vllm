#!/usr/bin/env bash
# Mirror vllm-project/vllm into upstream-main, and open or refresh one PR that
# brings it into dc-main. See upstream-sync.yml for why this lives on Buildkite.
#
# curl and python3 rather than the gh CLI, which hosted agents do not ship.
set -euo pipefail

UPSTREAM=${UPSTREAM:-https://github.com/vllm-project/vllm.git}
MIRROR=${MIRROR:-upstream-main}
OURS=${OURS:-dc-main}
REPO=${BUILDKITE_REPO_SLUG:-DaoCloud/vllm}
API=https://api.github.com

TOKEN=${UPSTREAM_SYNC_TOKEN:-}
if [ -z "$TOKEN" ] && command -v buildkite-agent >/dev/null 2>&1; then
  TOKEN=$(buildkite-agent secret get UPSTREAM_SYNC_TOKEN 2>/dev/null || true)
fi
[ -n "$TOKEN" ] || { echo "UPSTREAM_SYNC_TOKEN is not set"; exit 1; }

# Body on stdout; anything but 2xx prints what GitHub said and fails the build.
# Without this the script reported "refreshed PR #28" whether or not the call
# worked -- the PR sat two weeks out of date behind a green build.
api() {  # method path [json]
  local method=$1 path=$2 data=${3:-} out status body
  if [ -n "$data" ]; then
    out=$(curl -sS -w '\n%{http_code}' -X "$method" -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" -d "$data" "$API/$path")
  else
    out=$(curl -sS -w '\n%{http_code}' -X "$method" -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" "$API/$path")
  fi
  status=${out##*$'\n'}
  body=${out%$'\n'*}
  case "$status" in
    2*) printf '%s' "$body" ;;
    *)  { echo "GitHub API $method $path -> HTTP $status"
          printf '%s' "$body" | head -c 600; echo; } >&2
        return 1 ;;
  esac
}
# Tolerates empty input: when api() has already reported a failure, a decode
# traceback on top of it is noise that buries the message that matters.
jget() { python3 -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)
d = json.loads(raw)
print(eval(sys.argv[1], {}, {"d": d}) or "")' "$1"; }
jstr() { python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }

git config user.name  'buildkite[bot]'
git config user.email 'buildkite@users.noreply.github.com'

echo "--- fetching upstream"
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream "$UPSTREAM"
# Separately, deliberately: `fetch --tags` leaves FETCH_HEAD on whichever tag
# came last, and pushing that once put the mirror on a 0.17 release branch.
git fetch --quiet upstream main
git fetch --quiet --tags upstream
UP=$(git rev-parse upstream/main)
echo "upstream main ${UP:0:12}, ours $(git rev-parse --short HEAD)"

echo "--- mirroring into $MIRROR"
PUSH="https://x-access-token:${TOKEN}@github.com/${REPO}.git"
# Force: the mirror is defined as identical to upstream, not as a history of
# its own to protect.
git push --quiet --force "$PUSH" "upstream/main:refs/heads/${MIRROR}"
git push --quiet "$PUSH" 'refs/tags/*:refs/tags/*' 2>&1 | tail -3 || true
echo "$MIRROR = ${UP:0:12}, latest tag $(git tag --list 'v*' --sort=-creatordate | head -1)"

if git merge-base --is-ancestor upstream/main HEAD; then
  echo "--- $OURS already contains upstream; nothing to do"
  exit 0
fi
BEHIND=$(git rev-list --count "HEAD..upstream/main")

echo "--- testing the merge ($BEHIND commits behind)"
if git merge --no-commit --no-ff upstream/main >/dev/null 2>&1; then
  STATE=clean
else
  STATE=conflict
  CONFLICTS=$(git diff --name-only --diff-filter=U | head -40)
fi
git merge --abort 2>/dev/null || true

TITLE="Sync upstream into ${OURS} (${BEHIND} commits behind)"
BODY=$(printf '%s\n' \
  'Opened by the scheduled upstream sync.' '' \
  "\`${MIRROR}\` mirrors vllm-project/vllm@main." '' \
  'Merge with a **merge commit**. Squashing would make every later sync look' \
  'like a conflict; rebasing would rewrite the default branch under everyone.' '' \
  'What we changed relative to upstream, at any time:' '' \
  '```bash' \
  "git log --no-merges ${MIRROR}..${OURS}" \
  "git diff ${MIRROR}...${OURS}" \
  '```')

echo "--- opening or refreshing the sync PR"
OWNER=${REPO%%/*}
EXISTING=$(api GET "repos/${REPO}/pulls?state=open&base=${OURS}&head=${OWNER}:${MIRROR}" \
           | jget 'd[0]["number"] if d else ""')
jpayload() { python3 -c '
import json,sys
print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))
' "$@"; }
if [ -n "$EXISTING" ]; then
  api PATCH "repos/${REPO}/pulls/${EXISTING}" \
    "$(jpayload title "$TITLE" body "$BODY")" | jget 'd["html_url"]'
  echo "refreshed PR #${EXISTING}"
else
  api POST "repos/${REPO}/pulls" \
    "$(jpayload title "$TITLE" body "$BODY" head "$MIRROR" base "$OURS")" | jget 'd["html_url"]'
fi

[ "$STATE" = conflict ] || { echo "--- merge is clean"; exit 0; }

echo "--- reporting conflicts"
MARKER='<!-- upstream-sync-conflict -->'
IBODY=$(printf '%s\n' "$MARKER" \
  "Merging \`${MIRROR}\` into \`${OURS}\` conflicts. Upstream is ${BEHIND} commits ahead." '' \
  'Conflicting files:' '```' "$CONFLICTS" '```' '' \
  'To resolve:' '```bash' 'git fetch origin' \
  "git switch ${OURS} && git pull" "git merge origin/${MIRROR}" \
  '# fix, then' 'git commit && git push' '```' '' \
  'If upstream has absorbed one of our changes, `git revert` ours inside the' \
  'merge rather than deleting it, so the history says why it went away.')
# One thread reused, so a bad week is not seven issues.
FOUND=$(api GET "search/issues?q=$(python3 -c '
import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "repo:${REPO} is:issue is:open \"${MARKER}\"")" \
        | jget 'd["items"][0]["number"] if d.get("items") else ""')
if [ -n "$FOUND" ]; then
  api POST "repos/${REPO}/issues/${FOUND}/comments" \
    "{\"body\": $(printf '%s' "$IBODY" | jstr)}" | jget 'd.get("html_url")'
else
  api POST "repos/${REPO}/issues" \
    "{\"title\": \"Upstream sync conflicts with ${OURS}\", \"body\": $(printf '%s' "$IBODY" | jstr)}" \
    | jget 'd.get("html_url")'
fi
exit 1
