#!/bin/bash
# Re-seed everything the CN build needs into Nexus, for a given vLLM checkout.
#
# The build reaches outside for artefacts no CN mirror carries, on a link that
# cannot sustain a large transfer. Each one is copied into Nexus once; this
# script is how you refresh them after a version bump.
#
#   ./sync-cn-mirrors.sh /path/to/vllm            # everything
#   ./sync-cn-mirrors.sh /path/to/vllm wheels     # one group
#   groups: nexus | wheels | llguidance | nvshmem | git
#
# Every version is read out of the checkout -- requirements/cuda.txt,
# common.txt, CMakeLists.txt, cmake/external_projects/*.cmake,
# docker/Dockerfile -- so a bump needs no edit here. Re-running is safe:
# anything already in Nexus at the right version is skipped.
#
# Run it on the build box. Fetches that cross the border (the flashinfer
# wheels, the git mirrors) are the slow part; if this host's egress is having
# a bad day, fetch those on a machine with better connectivity and pass the
# directory as SEED_DIR=... to upload without re-downloading.
set -uo pipefail

VLLM=${1:?usage: $0 /path/to/vllm [wheels|llguidance|nvshmem|git]}
GROUP=${2:-all}

NEXUS=${NEXUS:-http://10.20.100.230:8081}
NEXUS_USER=${NEXUS_USER:-admin}
NEXUS_PW=${NEXUS_PW:-changeme}
PYPI_REPO=pypi-hosted
RAW_REPO=local-files
WORK=${WORK:-/var/tmp/cn-sync}
SEED_DIR=${SEED_DIR:-}
BUILD_IMAGE=${BUILD_IMAGE:-}

mkdir -p "$WORK"
ok=0; skip=0; fail=0
say() { printf '%s\n' "$*"; }
note_ok()   { ok=$((ok+1));   say "  ok      $*"; }
note_skip() { skip=$((skip+1)); say "  skip    $*"; }
note_fail() { fail=$((fail+1)); say "  FAIL    $*"; }

# Every slow thing here -- a 1.5 GB wheel, a cargo build, a mirror clone -- used
# to print only once it finished, which is indistinguishable from a hang. Say
# what is starting, then how it went.
step() { say "  ..      $*"; }
hhmmss() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

# curl -s on a multi-GB transfer is silent for minutes. Report progress while
# it runs so a stall is visible as one.
fetch_big() {
  local url=$1 out=$2 label=$3 t0 pid last now
  t0=$SECONDS
  curl -sL -C - --retry 20 --retry-delay 5 --retry-all-errors -o "$out" "$url" &
  pid=$!
  last=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(wc -c < "$out" 2>/dev/null | tr -d ' '); now=${now:-0}
    say "          $label $(( now / 1048576 )) MB  (+$(( (now - last) / 15 / 1024 )) KB/s)"
    last=$now
  done
  wait "$pid"; local rc=$?
  say "          $label 用时 $(hhmmss $(( SECONDS - t0 )))"
  return $rc
}

# Requirement lines are not uniform: some pin (`flashinfer-cubin==0.6.17`),
# some give a range with spaces (`llguidance >= 1.7.0, < 1.8.0`).
req()    { grep -hE "^$1 *[=><]" "$VLLM/requirements/cuda.txt" "$VLLM/requirements/common.txt" 2>/dev/null | head -1; }
pin_of() { req "$1" | grep -oE '==[0-9][0-9.]*' | head -1 | tr -d '='; }

# For a range, ask the index which version the build will actually pick.
resolve_of() {
  local spec
  spec=$(req "$1" | sed 's/;.*//' | tr -d ' ')
  [ -n "$spec" ] || return 1
  # uv prints the outgoing version too ("- llguidance==1.3.0" under Would
  # uninstall), and taking the first match picked that one -- a version the
  # spec does not even satisfy, which then failed the have_pypi check and sent
  # the script off to build a wheel it did not need.
  uv pip install --dry-run --no-deps --index-url "$NEXUS/repository/pypi-group/simple" "$spec" 2>&1 |
    grep -E '^ *\+' | grep -oE "$1==[0-9][0-9.]*" | head -1 | cut -d= -f3
}

cuda_tag() {  # 13.0.3 -> cu130
  local v
  v=$(grep -oE '^ARG CUDA_VERSION=[0-9.]+' "$VLLM/docker/Dockerfile" | head -1 | cut -d= -f2)
  printf 'cu%s' "$(echo "$v" | cut -d. -f1,2 | tr -d '.')"
}

# Returns 0 present, 1 absent, 2 could not tell. The third case matters: a 403
# from an un-accepted EULA, or any blip, used to look exactly like "absent" and
# start a 1.5 GB download of a file already sitting in Nexus.
have_pypi() {  # name version-or-filename
  local body code
  body=$(curl -s -m 30 -w '\n%{http_code}' "$NEXUS/repository/$PYPI_REPO/simple/$1/" 2>/dev/null)
  code=${body##*$'\n'}
  [ "$code" = 200 ] || return 2
  printf '%s' "${body%$'\n'*}" | grep -qF "$2"
}

upload_pypi() {
  local f=$1 code
  code=$(curl -s -u "$NEXUS_USER:$NEXUS_PW" -X POST \
    "$NEXUS/service/rest/v1/components?repository=$PYPI_REPO" \
    -F "pypi.asset=@$f" -o /dev/null -w '%{http_code}')
  [[ $code == 2* ]]
}

# ---------------------------------------------------------------- flashinfer
# Published only as GitHub release assets: no CN mirror carries them, and uv
# installs a batch atomically, so one dropped connection discards the lot.
sync_wheels() {
  local ver tag base f url
  ver=$(pin_of flashinfer-cubin)
  tag=$(cuda_tag)
  [ -n "$ver" ] || { note_fail "flashinfer: no pin found in requirements"; return; }
  base="https://github.com/flashinfer-ai/flashinfer/releases/download/v$ver"
  say "flashinfer $ver ($tag)"

  for f in "flashinfer_cubin-$ver-py3-none-any.whl" \
           "flashinfer_jit_cache-$ver+$tag-cp39-abi3-manylinux_2_28_x86_64.whl"; do
    local name=${f%%-*}; name=${name//_/-}
    have_pypi "$name" "$f"
    case $? in
      0) note_skip "$f"; continue ;;
      2) note_fail "$f (无法查询 $PYPI_REPO 索引 -- 不下载，先修好 Nexus 再重跑)"; continue ;;
    esac
    url="$base/$f"
    if [ -n "$SEED_DIR" ] && [ -f "$SEED_DIR/$f" ]; then
      step "$f 从 SEED_DIR 复制"
      cp "$SEED_DIR/$f" "$WORK/$f"
    else
      step "$f 跨境下载中（约 1 GB，-C - 断点续传）"
      fetch_big "$url" "$WORK/$f" "$f" || { note_fail "$f (download)"; continue; }
    fi
    step "$f 校验压缩包"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$WORK/$f" 2>/dev/null || {
      note_fail "$f (corrupt archive)"; rm -f "$WORK/$f"; continue; }
    step "$f 上传 Nexus"
    upload_pypi "$WORK/$f" && note_ok "$f" || note_fail "$f (upload)"
  done
}

# ---------------------------------------------------------------- llguidance
# Ships manylinux_2_31 wheels only; the build image is glibc 2.28, so uv falls
# back to the sdist, whose build bootstraps Rust from static.rust-lang.org.
# Build it once here against CN Rust mirrors and publish the wheel.
sync_llguidance() {
  local ver img out
  ver=$(pin_of llguidance || true); [ -n "$ver" ] || ver=$(resolve_of llguidance)
  [ -n "$ver" ] || { note_fail "llguidance: no pin found"; return; }
  say "llguidance $ver"
  have_pypi llguidance "$ver"
  case $? in
    0) note_skip "llguidance-$ver"; return ;;
    2) note_fail "llguidance (无法查询 $PYPI_REPO 索引 -- 不构建，先修好 Nexus 再重跑)"; return ;;
  esac

  # The build needs docker and the manylinux build image. On a laptop that is a
  # ~6 GB pull started silently, so refuse instead.
  if ! docker info >/dev/null 2>&1; then
    note_fail "llguidance (需要 docker -- 在构建机上跑这一组)"; return
  fi

  img=${BUILD_IMAGE:-$(grep -oE '^ARG BUILD_BASE_IMAGE=.*' "$VLLM/docker/Dockerfile" | head -1 | cut -d= -f2-)}
  case "$img" in */*/*) ;; *) img="release.daocloud.io/docker-1ms/$img" ;; esac
  out=$WORK/llg; rm -rf "$out"; mkdir -p "$out"
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    note_fail "llguidance (本机没有 $img -- 在构建机上跑，别在这里拉 6 GB)"; return
  fi
  step "llguidance $ver 源码构建中（容器内，日志 $WORK/llg-build.log）"

  docker run --rm -v "$out":/out \
    -e RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
    -e RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup \
    -e RUSTUP_TOOLCHAIN=stable \
    -e PIP_INDEX_URL="$NEXUS/repository/pypi-group/simple" \
    -e PIP_TRUSTED_HOST="${NEXUS#http://}" \
    -e LLG_VER="$ver" \
    "$img" bash -c '
      set -euo pipefail
      export RUSTUP_HOME=/tmp/rustup CARGO_HOME=/tmp/cargo
      mkdir -p "$CARGO_HOME"
      curl -sSfL -o /tmp/rustup-init \
        "$RUSTUP_DIST_SERVER/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
      chmod +x /tmp/rustup-init
      /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain stable
      export PATH="$CARGO_HOME/bin:$PATH"
      printf "[source.crates-io]\nreplace-with = \"tuna\"\n[source.tuna]\nregistry = \"sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/\"\n" \
        > "$CARGO_HOME/config.toml"
      /opt/python/cp312-cp312/bin/python -m pip wheel \
        --trusted-host "${PIP_TRUSTED_HOST%%:*}" --no-deps "llguidance==$LLG_VER" -w /out
    ' >"$WORK/llg-build.log" 2>&1 || { note_fail "llguidance (build; see $WORK/llg-build.log)"; return; }

  local whl; whl=$(find "$out" -name '*.whl' | head -1)
  [ -n "$whl" ] || { note_fail "llguidance (no wheel produced)"; return; }
  upload_pypi "$whl" && note_ok "$(basename "$whl")" || note_fail "$(basename "$whl") (upload)"
}

# ------------------------------------------------------------------- nvshmem
# 169 MB from developer.download.nvidia.com; the installer's curl --retry
# restarts rather than resumes, so a mid-file drop is fatal.
sync_nvshmem() {
  local ver tag file url dest
  # The Dockerfile passes NVSHMEM_VER through without a default; the real
  # default lives in the installer script.
  ver=$(grep -oE '^ARG NVSHMEM_VER=[0-9][0-9.]*' "$VLLM/docker/Dockerfile" | head -1 | cut -d= -f2)
  [ -n "$ver" ] || ver=$(grep -oE 'NVSHMEM_VER:-"?[0-9][0-9.]*' "$VLLM/tools/ep_kernels/install_python_libraries.sh" |
                         head -1 | grep -oE '[0-9][0-9.]*')
  [ -n "$ver" ] || { note_fail "nvshmem: no version found"; return; }
  tag=$(cuda_tag); tag=${tag#cu}; tag="cuda${tag:0:2}"
  file="libnvshmem-linux-x86_64-${ver}_${tag}-archive.tar.xz"
  dest="$NEXUS/repository/$RAW_REPO/nvshmem/$file"
  say "nvshmem $ver"

  if [ "$(curl -s -o /dev/null -m 30 -w '%{http_code}' "$dest")" = 200 ]; then
    note_skip "$file"; return
  fi
  url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/$file"
  if [ -n "$SEED_DIR" ] && [ -f "$SEED_DIR/$file" ]; then
    cp "$SEED_DIR/$file" "$WORK/$file"
  else
    curl -sL -C - --retry 20 --retry-delay 5 --retry-all-errors -o "$WORK/$file" "$url" || {
      note_fail "$file (download)"; return; }
  fi
  tar -tJf "$WORK/$file" >/dev/null 2>&1 || { note_fail "$file (corrupt archive)"; return; }
  curl -s -u "$NEXUS_USER:$NEXUS_PW" --upload-file "$WORK/$file" "$dest" -o /dev/null -w '' \
    && note_ok "$file" || note_fail "$file (upload)"
}

# ----------------------------------------------------------------------- git
# cmake FetchContent and the DeepEP clone all target github.com. Mirrors are
# published as dumb-HTTP bare repos under a path identical to the GitHub
# suffix, so one insteadOf in the build rewrites every one of them.
#
# The path must match the URL the build uses exactly, .git suffix included --
# FlashMLA is referenced without it, the rest with it.
# Emit "owner/repo<TAB>ref" for everything the build clones. The ref matters:
# after a version bump the mirror may be served fine yet lack the new commit,
# and a stale mirror fails later as "reference is not a tree", far from here.
git_targets() {
  python3 - "$VLLM" <<'PYEOF'
import pathlib, re, sys

root = pathlib.Path(sys.argv[1])
out = []

def cmake_blocks(text):
    # Both spellings appear: _Declare for most, _Populate for deepgemm/qutlass.
    for m in re.finditer(r'FetchContent_(?:Declare|Populate)\s*\((.*?)\n\s*\)', text, re.S):
        yield m.group(1)

vars_ = {}
for f in sorted((root / "cmake/external_projects").glob("*.cmake")) + [root / "CMakeLists.txt"]:
    if not f.exists():
        continue
    text = f.read_text()
    # A CUDA build never takes the ROCm branch of the triton fork selector.
    text = re.sub(r'if\s*\(\s*VLLM_TARGET_DEVICE STREQUAL "rocm"\s*\).*?else\s*\(\s*\)', '', text, flags=re.S)
    for m in re.finditer(r'set\(\s*(\w+)\s+"?(https://github\.com/[^"\s)]+|[0-9a-f]{40}|v[0-9][^"\s)]*)"?\s*\)', text):
        vars_[m.group(1)] = m.group(2)
    for blk in cmake_blocks(text):
        repo = re.search(r'GIT_REPOSITORY\s+"?([^"\s]+)"?', blk)
        tag = re.search(r'GIT_TAG\s+"?([^"\s]+)"?', blk)
        if not repo:
            continue
        r, t = repo.group(1), tag.group(1) if tag else ""
        for name, val in vars_.items():
            r = r.replace("${%s}" % name, val)
            t = t.replace("${%s}" % name, val)
        # Whatever GIT_SUBMODULES says is exactly what cmake will initialise,
        # so it is also all that needs mirroring. Without it the walk pulls
        # every submodule in .gitmodules -- for flash-attention that is two
        # ROCm repos a CUDA build never compiles, measured at 565 MB.
        sub = re.search(r'GIT_SUBMODULES\s+([^\n]*)', blk)
        subs = " ".join(a or b for a, b in re.findall(r'"([^"]+)"|(\S+)', sub.group(1))) if sub else ""
        if r.startswith("https://github.com/"):
            out.append((r, t, subs))

# The ep_kernels installer clones DeepEP directly, pinned by a Dockerfile ARG.
inst = root / "tools/ep_kernels/install_python_libraries.sh"
if inst.exists():
    for m in re.finditer(r'"(https://github\.com/[^"]+)"', inst.read_text()):
        ref = ""
        d = (root / "docker/Dockerfile")
        if d.exists():
            a = re.search(r'^ARG DEEPEP_COMMIT_HASH=(\S+)', d.read_text(), re.M)
            ref = a.group(1) if a else ""
        out.append((m.group(1), ref, ""))

# Two crates come from GitHub over git rather than the registry; the fragment
# after '#' is the commit cargo actually resolves to.
lock = root / "rust/Cargo.lock"
if lock.exists():
    for m in re.finditer(
        r'source = "git\+(https://github\.com/[^?#"]+?)(?:\.git)?(?:\?[^"#]*)?#([0-9a-f]{40})"',
        lock.read_text(),
    ):
        out.append((m.group(1), m.group(2), ""))

# The build patches flash-attention the same way deepgemm.cmake and
# flashkda.cmake already declare it; keep the two in step.
NARROW = {"vllm-project/flash-attention.git": "csrc/cutlass"}

seen = set()
for url, ref, subs in out:
    suffix = url[len("https://github.com/"):]
    if suffix in seen:
        continue
    seen.add(suffix)
    print("%s\t%s\t%s" % (suffix, ref, subs or NARROW.get(suffix, "")))
PYEOF
}

# Mirror one repo and publish it under every spelling the build may use.
# GitHub is case-insensitive, Nexus raw paths are not.
mirror_one() {
  local suffix=$1 ref=$2 src bare probe base n rel alias
  base=${suffix%.git}
  bare="$WORK/git/$base.git"
  probe="$NEXUS/repository/$RAW_REPO/git/$suffix/info/refs"

  # Served is not enough: the pinned ref has to be in there. Two ways to tell,
  # and both are needed. info/refs lists ref tips, which covers a tag or a
  # branch head; a submodule pins an arbitrary commit that is usually an
  # ancestor of one, invisible there and unreachable over dumb HTTP once the
  # objects are packed. For those, ask the bare clone this box published from.
  if [ -n "$ref" ] && [ "$(curl -s -o /dev/null -m 30 -w '%{http_code}' "$probe")" = 200 ]; then
    if curl -s -m 60 "$probe" | grep -qiF "$ref" ||
       git ls-remote "$NEXUS/repository/$RAW_REPO/git/$suffix" 2>/dev/null | grep -qiF "$ref" ||
       ( [ -d "$bare" ] && cd "$bare" && git cat-file -e "$ref^{commit}" 2>/dev/null ); then
      note_skip "$suffix @ ${ref:0:12}"
      return 0
    fi
    say "  stale   $suffix (missing ${ref:0:12}) -- refreshing"
  fi

  src="https://github.com/$base.git"
  case "$base" in triton-lang/triton) src="https://gitee.com/triton-lang/triton.git" ;; esac
  local t0=$SECONDS
  if [ -d "$bare" ]; then
    step "$suffix 增量 fetch"
    ( cd "$bare" && git remote set-url origin "$src" && git fetch -q --prune origin '+refs/*:refs/*' ) ||
      { note_fail "$suffix (fetch)"; return 1; }
  else
    # A first clone of cutlass is ~400 MB over the border. Say so before the
    # silence starts, and say where it is going.
    step "$suffix 首次完整克隆 from $src（可能很慢）"
    mkdir -p "$(dirname "$bare")"
    git clone --mirror "$src" "$bare" >/dev/null 2>&1 || { note_fail "$suffix (clone from $src)"; return 1; }
  fi
  step "$suffix repack + update-server-info"
  ( cd "$bare" && git repack -a -d -q && git update-server-info ) || { note_fail "$suffix (repack)"; return 1; }
  step "$suffix 上传 $(du -sh "$bare" | cut -f1) 到 Nexus，用时 $(hhmmss $(( SECONDS - t0 )))"

  # Publish under the exact suffix asked for, plus the other spellings that
  # appear across vLLM and its submodules.
  for alias in "$base.git" "$base" "$(echo "$base" | sed 's#^nvidia/#NVIDIA/#').git" "$(echo "$base" | sed 's#^NVIDIA/#nvidia/#').git"; do
    case " $published " in *" $alias "*) continue ;; esac
    n=0
    while IFS= read -r rel; do
      rel=${rel#./}
      curl -s -u "$NEXUS_USER:$NEXUS_PW" --upload-file "$bare/$rel" \
        "$NEXUS/repository/$RAW_REPO/git/$alias/$rel" -o /dev/null && n=$((n+1))
    done < <(cd "$bare" && find . -type f)
    published="$published $alias"
  done
  note_ok "$suffix @ ${ref:0:12} ($(du -sh "$bare" | cut -f1))"
}

# Submodules resolve through the same rewrite, so they need mirrors too, at
# the commit the parent pins -- and they can appear or vanish between versions.
# Emits "owner/repo<TAB>sha". The sha matters: mirror_one only skips a repo
# whose pinned ref it can find, so handing it an empty ref made every sync
# re-clone every submodule -- cutlass alone is ~400 MB over a link that cannot
# spare it. The parent tree records the exact commit, so read it.
# Emits "owner/repo<TAB>sha" for the submodules the build will actually
# initialise. The sha matters: mirror_one only skips a repo whose pinned ref it
# can find, so handing it an empty ref made every sync re-clone every
# submodule. The path filter matters just as much -- flash-attention lists two
# ROCm repos a CUDA build never compiles, 565 MB pulled across the border for
# nothing. $3 is the parent's GIT_SUBMODULES; empty means take them all.
submodules_of() {
  local base=$1 ref=$2 want=${3:-} bare="$WORK/git/${1%.git}.git"
  [ -d "$bare" ] || return 0
  ( cd "$bare" &&
    git show "${ref:-HEAD}:.gitmodules" 2>/dev/null |
      awk '/path *=/{sub(/.*= */,""); p=$0}
           /url *=/ {sub(/.*= */,""); u=$0}
           p && u   {print p "\t" u; p=""; u=""}' |
      while IFS=$'\t' read -r path url; do
        case "$url" in https://github.com/*) ;; *) continue ;; esac
        if [ -n "$want" ]; then
          case " $want " in *" $path "*) ;; *) continue ;; esac
        fi
        sha=$(git ls-tree "${ref:-HEAD}" "$path" | awk '$2 == "commit" {print $3}')
        printf '%s\t%s\n' "${url#https://github.com/}" "$sha"
      done
  ) | sort -u
}

sync_git() {
  say "git mirrors"
  published=""
  local suffix ref sub
  while IFS=$'\t' read -r suffix ref subs; do
    [ -n "$suffix" ] || continue
    mirror_one "$suffix" "$ref" || continue
    # One level of submodules covers what vLLM's dependencies actually declare.
    while IFS=$'\t' read -r sub sub_ref; do
      [ -n "$sub" ] && mirror_one "$sub" "$sub_ref"
    done < <(submodules_of "$suffix" "$ref" "$subs")
  done < <(git_targets)
}

# Nexus 3.69 OSS has no Cargo format, so the Rust side is three raw repos
# standing in for one. The sparse-index protocol is plain HTTP GET, which a
# raw proxy serves fine; the single thing it cannot serve is index/config.json,
# because upstream's copy names rsproxy.cn as the download base and would send
# cargo straight back out. A hosted repo holding just that file, ordered first
# in a group, overrides it.
#
# This is the only place the upstream Rust mirror is named. The build itself
# only ever sees $NEXUS/repository/cargo.
CARGO_UPSTREAM=${CARGO_UPSTREAM:-https://rsproxy.cn}

# dnf pulls ccache, git, curl, sudo, rdma-core-devel, make and numactl-devel
# from AlmaLinux and EPEL. Proxies are rooted at the distro rather than a point
# release so the repo files can leave $releasever to dnf, and a base image
# bumped to another AlmaLinux 8.x keeps working without touching anything here.
RPM_UPSTREAM=${RPM_UPSTREAM:-https://mirrors.aliyun.com}

nexus_yum_repo() {
  local name=$1 url=$2 code
  if [ "$(curl -s -o /dev/null -m 30 -u "$NEXUS_USER:$NEXUS_PW" -w '%{http_code}' \
          "$NEXUS/service/rest/v1/repositories/yum/proxy/$name")" = 200 ]; then
    note_skip "repo $name"
    return 0
  fi
  code=$(curl -s -u "$NEXUS_USER:$NEXUS_PW" -X POST \
    "$NEXUS/service/rest/v1/repositories/yum/proxy" \
    -H 'Content-Type: application/json' -o /tmp/nexus-yum.out -w '%{http_code}' -d "{
      \"name\":\"$name\",\"online\":true,
      \"storage\":{\"blobStoreName\":\"default\",\"strictContentTypeValidation\":false},
      \"proxy\":{\"remoteUrl\":\"$url\",\"contentMaxAge\":-1,\"metadataMaxAge\":1440},
      \"negativeCache\":{\"enabled\":true,\"timeToLive\":1},
      \"httpClient\":{\"blocked\":false,\"autoBlock\":true},
      \"yumSigning\":{}}")
  case "$code" in
    201) note_ok "repo $name" ;;
    *)   note_fail "repo $name (http=$code $(head -c 120 /tmp/nexus-yum.out))" ;;
  esac
}

# Ask the base image what it is rather than pinning a release here.
rpm_release() {
  local img
  img=$(grep -oE '^ARG BUILD_BASE_IMAGE=.*' "$VLLM/docker/Dockerfile" | head -1 | cut -d= -f2-)
  case "$img" in */*/*) ;; *) img="release.daocloud.io/docker-1ms/$img" ;; esac
  # Only ask the image if it is already here. Off the build box this would
  # otherwise start a ~6 GB pull just to read one line of /etc/os-release.
  if docker image inspect "$img" >/dev/null 2>&1; then
    docker run --rm --entrypoint sh "$img" -c '. /etc/os-release; echo "$VERSION_ID"' 2>/dev/null && return
  fi
  echo "${RPM_RELEASE:-8.10}"
}
rpm_major() { rpm_release | cut -d. -f1; }

nexus_repo() {
  local kind=$1 name=$2 body=$3 code
  if [ "$(curl -s -o /dev/null -m 30 -w '%{http_code}' -u "$NEXUS_USER:$NEXUS_PW" \
          "$NEXUS/service/rest/v1/repositories/raw/$kind/$name")" = 200 ]; then
    note_skip "repo $name"
    return 0
  fi
  code=$(curl -s -u "$NEXUS_USER:$NEXUS_PW" -X POST \
    "$NEXUS/service/rest/v1/repositories/raw/$kind" \
    -H 'Content-Type: application/json' -d "$body" -o /tmp/nexus-repo.out -w '%{http_code}')
  case "$code" in
    201) note_ok "repo $name" ;;
    *)   note_fail "repo $name (http=$code $(head -c 120 /tmp/nexus-repo.out))" ;;
  esac
}

sync_nexus() {
  say "nexus repositories"
  nexus_repo proxy cargo-proxy "{
    \"name\":\"cargo-proxy\",
    \"online\":true,
    \"storage\":{\"blobStoreName\":\"default\",\"strictContentTypeValidation\":false},
    \"proxy\":{\"remoteUrl\":\"$CARGO_UPSTREAM\",\"contentMaxAge\":1440,\"metadataMaxAge\":1440},
    \"negativeCache\":{\"enabled\":true,\"timeToLive\":1},
    \"httpClient\":{\"blocked\":false,\"autoBlock\":true},
    \"raw\":{\"contentDisposition\":\"ATTACHMENT\"}
  }"
  nexus_repo hosted cargo-hosted '{
    "name":"cargo-hosted",
    "online":true,
    "storage":{"blobStoreName":"default","strictContentTypeValidation":false,"writePolicy":"ALLOW"},
    "raw":{"contentDisposition":"ATTACHMENT"}
  }'
  nexus_repo group cargo '{
    "name":"cargo",
    "online":true,
    "storage":{"blobStoreName":"default","strictContentTypeValidation":false},
    "group":{"memberNames":["cargo-hosted","cargo-proxy"]},
    "raw":{"contentDisposition":"ATTACHMENT"}
  }'

  # cargo appends /{crate}/{version}/download to "dl".
  local want dl
  dl="$NEXUS/repository/cargo/api/v1/crates"
  want="{\"dl\":\"$dl\",\"api\":\"$CARGO_UPSTREAM\"}"
  if [ "$(curl -s -m 30 "$NEXUS/repository/cargo/index/config.json" | tr -d ' \n')" = "$want" ]; then
    note_skip "cargo index/config.json"
  else
    printf '%s\n' "$want" > "$WORK/config.json"
    if curl -s -u "$NEXUS_USER:$NEXUS_PW" -X POST \
        "$NEXUS/service/rest/v1/components?repository=cargo-hosted" \
        -F "raw.directory=/index" -F "raw.asset1=@$WORK/config.json" \
        -F "raw.asset1.filename=config.json" -o /dev/null -w '%{http_code}' | grep -q '20'; then
      note_ok "cargo index/config.json"
    else
      note_fail "cargo index/config.json"
    fi
  fi

  nexus_yum_repo almalinux "$RPM_UPSTREAM/almalinux/"
  nexus_yum_repo epel      "$RPM_UPSTREAM/epel/"

  # Prove the group actually answers on all three paths the build uses.
  local probe
  local probe
  for probe in "cargo/index/config.json" \
               "cargo/index/an/yh/anyhow" \
               "cargo/rustup/dist/x86_64-unknown-linux-gnu/rustup-init" \
               "almalinux/$(rpm_release)/BaseOS/x86_64/os/repodata/repomd.xml" \
               "almalinux/$(rpm_release)/AppStream/x86_64/os/repodata/repomd.xml" \
               "epel/$(rpm_major)/Everything/x86_64/repodata/repomd.xml"; do
    step "探测 $probe"
    if [ "$(curl -sL -o /dev/null -m 180 -w '%{http_code}' "$NEXUS/repository/$probe")" = 200 ]; then
      note_ok "serves $probe"
    else
      note_fail "serves $probe"
    fi
  done
}

say "syncing into $NEXUS from $VLLM"
say "工作目录 $WORK  （git 裸库缓存在这里；换机器跑会重新完整克隆）"
case "$GROUP" in
  all)        sync_nexus; sync_wheels; sync_llguidance; sync_nvshmem; sync_git ;;
  nexus)      sync_nexus ;;
  wheels)     sync_wheels ;;
  llguidance) sync_llguidance ;;
  nvshmem)    sync_nvshmem ;;
  git)        sync_git ;;
  *) say "unknown group: $GROUP"; exit 2 ;;
esac

say "done: $ok synced, $skip already current, $fail failed"
[ "$fail" -eq 0 ]
