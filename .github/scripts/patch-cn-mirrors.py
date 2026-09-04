#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Point the build at the CN mirrors, at build time rather than in the tree.

Kept out of docker/Dockerfile and requirements/ so those stay byte-identical to
upstream and merges stay clean. Run from the repo root before `docker build`.
"""

import pathlib
import re
import sys

NEXUS = "http://10.20.100.230:8081/repository/pypi-group/simple"

# Docker scopes ARG per stage. The global PIP_INDEX_URL / UV_INDEX_URL block
# sits above the first FROM, so `base` never saw it and uv there resolved
# against the default index no matter what --build-arg said -- which is why
# flashinfer-cubin==0.6.17 came back "no version" even with the wheel sitting
# in the local index. Re-declare as ENV, which the stages built FROM base
# inherit, placed after the dnf and sccache layers so those stay cached.
# vllm-runtime-base starts from a different image and needs its own copy.
DECL = (
    "ARG PIP_INDEX_URL\n"
    "ARG PIP_EXTRA_INDEX_URL\n"
    "ARG UV_INDEX_URL=${PIP_INDEX_URL}\n"
    "ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}\n"
    "ENV UV_INDEX_URL=${UV_INDEX_URL}\n"
    "ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}\n"
    # uv has deadlocked three times on this box -- every thread parked on a
    # futex, no CPU, writes frozen, at 65 MB one run and 10.5 GB another. It
    # defaults its pools to the core count, which is 96 here. Cap them.
    "ENV UV_CONCURRENT_DOWNLOADS=4\n"
    "ENV UV_CONCURRENT_INSTALLS=8\n"
    "ENV UV_CONCURRENT_BUILDS=4\n"
    # setup_deepgemm_pythons.sh provisions one interpreter per requires-python
    # entry -- 3.10 through 3.14, ~30 MB each -- and uv fetches them from
    # python-build-standalone's GitHub releases. Nexus cannot stand in: it
    # indexes PyPI, not interpreters. A CN mirror installs one in under 7 s.
    "ENV UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone\n"
)

UV_ANCHOR = (
    "# Install uv and bootstrap /opt/venv from the Python interpreter included in"
)

# Every git fetch in the build -- the ep_kernels DeepEP clone and each cmake
# FetchContent -- targets https://github.com/..., which this network cannot
# sustain. The mirrors live as dumb-HTTP bare repos under Nexus local-files,
# path-identical to the GitHub suffix, so one insteadOf rewrites them all.
GIT_REDIRECT = (
    'RUN git config --system url."http://10.20.100.230:8081/repository/local-files/git/"'
    '.insteadOf "https://github.com/"\n'
)

RT_ANCHOR = "FROM ${FINAL_BASE_IMAGE} AS vllm-runtime-base"

NEXUS_HOST = "http://10.20.100.230:8081"

# The three dnf layers all sit in base or a stage built FROM it, so rewriting
# /etc/yum.repos.d once in base covers every one. $releasever is left for dnf
# to substitute, which keeps this working when the base image moves to another
# AlmaLinux point release -- the proxies are rooted at the distro, not a
# version.
DNF_ANCHOR = "# Install system dependencies including build tools. The PyTorch manylinux"
DNF_REDIRECT = (
    "RUN rm -f /etc/yum.repos.d/*.repo \\\n"
    "    && printf '%s\\n' \\\n"
    "        '[baseos]' 'name=AlmaLinux $releasever - BaseOS' \\\n"
    f"        'baseurl={NEXUS_HOST}/repository/almalinux/$releasever/BaseOS/$basearch/os/' \\\n"
    "        'gpgcheck=0' 'enabled=1' \\\n"
    "        '[appstream]' 'name=AlmaLinux $releasever - AppStream' \\\n"
    f"        'baseurl={NEXUS_HOST}/repository/almalinux/$releasever/AppStream/$basearch/os/' \\\n"
    "        'gpgcheck=0' 'enabled=1' \\\n"
    "        '[extras]' 'name=AlmaLinux $releasever - Extras' \\\n"
    f"        'baseurl={NEXUS_HOST}/repository/almalinux/$releasever/extras/$basearch/os/' \\\n"
    "        'gpgcheck=0' 'enabled=1' \\\n"
    "        '[epel]' 'name=EPEL $releasever' \\\n"
    f"        'baseurl={NEXUS_HOST}/repository/epel/$releasever/Everything/$basearch/' \\\n"
    "        'gpgcheck=0' 'enabled=1' \\\n"
    "        > /etc/yum.repos.d/cn.repo\n"
    "\n"
)

CARGO_REPO = "http://10.20.100.230:8081/repository/cargo"

# build_rust.sh bootstraps rustup from sh.rustup.rs, pulls the pinned toolchain
# from static.rust-lang.org, then resolves every crate against crates.io. All
# three cross the border, and they are what killed the run that had already
# finished compiling CUDA. Nexus 3.69 OSS has no Cargo format, but the sparse
# index is plain HTTP GET, so a raw group in front of rsproxy serves the index,
# the .crate files, rustup itself and the toolchain alike.
#
# Installing rustup here rather than letting build_rust.sh do it makes this a
# cached layer keyed on rust-toolchain.toml, and leaves build_rust.sh finding a
# toolchain already in place, so it skips both of its downloads.
#
# git-fetch-with-cli matters: two crates come from GitHub over git, and cargo's
# built-in libgit2 ignores the system-wide insteadOf redirect that every other
# fetch in this build relies on. Shelling out to git honours it.
RUST_ANCHOR = (
    "# Copy only the Rust build inputs; build_rust.sh publishes artifacts needed"
)
RUST_SETUP = (
    f"ENV RUSTUP_DIST_SERVER={CARGO_REPO}\n"
    f"ENV RUSTUP_UPDATE_ROOT={CARGO_REPO}/rustup\n"
    "ENV CARGO_HOME=/root/.cargo\n"
    "ENV RUSTUP_HOME=/root/.rustup\n"
    "ENV PATH=/root/.cargo/bin:$PATH\n"
    "COPY rust-toolchain.toml rust-toolchain.toml\n"
    "RUN mkdir -p ${CARGO_HOME} \\\n"
    "    && printf '%s\\n' '[source.crates-io]' 'replace-with = \"nexus\"' "
    f"'[source.nexus]' 'registry = \"sparse+{CARGO_REPO}/index/\"' "
    "'[net]' 'git-fetch-with-cli = true' > ${CARGO_HOME}/config.toml \\\n"
    # rustup-init dispatches on argv[0]; any other name and it looks for a
    # toolchain proxy by that name and exits.
    '    && curl -sSfL -o /tmp/rustup-init "${RUSTUP_UPDATE_ROOT}/dist/x86_64-unknown-linux-gnu/rustup-init" \\\n'
    "    && chmod +x /tmp/rustup-init \\\n"
    "    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain none \\\n"
    "    && rm /tmp/rustup-init \\\n"
    "    && rustup toolchain install \"$(tr -d ' \"' < rust-toolchain.toml | sed -n 's/^channel=//p')\"\n"
    "\n"
)


def stage_parents(dockerfile: str) -> dict[str, str]:
    """Map each stage to the stage it is built FROM, where that is one."""
    names, parents = set(), {}
    for line in dockerfile.splitlines():
        if line.startswith("FROM ") and " AS " in line:
            base, name = line[5:].rsplit(" AS ", 1)
            base, name = base.strip(), name.strip()
            if base in names:
                parents[name] = base
            names.add(name)
    return parents


def stage_has(dockerfile: str, stage: str, prefix: str) -> bool:
    """Whether the stage declares the line, or inherits it from an ancestor.

    ENV crosses a FROM, so asking only about the named stage gives the wrong
    answer: upstream split the Rust build into rust-build-cache and a
    rust-build built FROM it, and the settings that land in the parent are in
    force in the child even though the child never mentions them.
    """
    parents = stage_parents(dockerfile)
    wanted, seen = set(), set()
    while stage and stage not in seen:
        seen.add(stage)
        wanted.add(stage)
        stage = parents.get(stage)

    current = None
    for line in dockerfile.splitlines():
        if line.startswith("FROM ") and " AS " in line:
            current = line.rsplit(" AS ", 1)[1].strip()
        elif current in wanted and line.startswith(prefix):
            return True
    return False


def main() -> int:
    df = pathlib.Path("docker/Dockerfile")
    rq = pathlib.Path("requirements/cuda.txt")
    dockerfile, requirements = df.read_text(), rq.read_text()

    for anchor, replacement in (
        (UV_ANCHOR, DECL + GIT_REDIRECT + UV_ANCHOR),
        (RT_ANCHOR, RT_ANCHOR + "\n" + DECL),
        (RUST_ANCHOR, RUST_SETUP + RUST_ANCHOR),
        (DNF_ANCHOR, DNF_REDIRECT + DNF_ANCHOR),
    ):
        seen = dockerfile.count(anchor)
        if seen != 1:
            print(
                f"anchor appears {seen} times, expected 1: {anchor[:60]}",
                file=sys.stderr,
            )
            return 1
        dockerfile = dockerfile.replace(anchor, replacement, 1)

    # flashinfer-cubin and flashinfer-jit-cache are published only as GitHub
    # release assets, and uv installs a batch atomically: one dropped connection
    # discards every byte already fetched, so a 1 GB wheel never lands on a
    # link that stutters. Both are seeded into the local index instead.
    requirements = re.sub(
        r"^--extra-index-url https://flashinfer\.ai/whl/\n",
        "",
        requirements,
        flags=re.M,
    )
    dockerfile = re.sub(
        r"--index-url https://flashinfer\.ai/whl/cu[^\n]*",
        "--index-url " + NEXUS,
        dockerfile,
    )

    # The ep_kernels stage pulls NVSHMEM from developer.download.nvidia.com,
    # which drops mid-file here (curl 18, no resume). The archive is seeded in
    # Nexus; same bytes, LAN speed.
    ep = pathlib.Path("tools/ep_kernels/install_python_libraries.sh")
    eps = ep.read_text()
    nv_old = 'NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_SUBDIR}/${NVSHMEM_FILE}"'
    if nv_old not in eps:
        sys.exit("NVSHMEM_URL anchor missing in install_python_libraries.sh")
    eps = eps.replace(
        nv_old,
        'NVSHMEM_URL="http://10.20.100.230:8081/repository/local-files/nvshmem/${NVSHMEM_FILE}"',
        1,
    )
    ep.write_text(eps)

    # flash-attention declares no GIT_SUBMODULES, so cmake initialises all
    # three -- including two ROCm repos a CUDA build never compiles, ~600 MB
    # to mirror for nothing. Name the one that is actually needed, matching
    # what deepgemm.cmake and flashkda.cmake already do.
    fa = pathlib.Path("cmake/external_projects/vllm_flash_attn.cmake")
    fas = fa.read_text()
    m = re.search(
        r"GIT_REPOSITORY https://github\.com/vllm-project/flash-attention\.git\n"
        r"(\s*)GIT_TAG \S+",
        fas,
    )
    if not m:
        sys.exit("flash-attention FetchContent block not recognised")
    fa_anchor = m.group(0).splitlines()[-1].strip()
    if "GIT_SUBMODULES" not in fas:
        indent = fas[: fas.index(fa_anchor)].rsplit("\n", 1)[-1]
        fas = fas.replace(
            fa_anchor, fa_anchor + "\n" + indent + "GIT_SUBMODULES csrc/cutlass", 1
        )
        fa.write_text(fas)

    # The git mirrors speak dumb HTTP, which cannot serve shallow clones.
    cm = pathlib.Path("CMakeLists.txt")
    cms = cm.read_text()
    if "GIT_SHALLOW TRUE" not in cms:
        sys.exit("cutlass GIT_SHALLOW anchor missing in CMakeLists.txt")
    cms = cms.replace("GIT_SHALLOW TRUE", "GIT_SHALLOW FALSE", 1)
    cm.write_text(cms)

    df.write_text(dockerfile)
    rq.write_text(requirements)

    if "GIT_SUBMODULES csrc/cutlass" not in fa.read_text():
        print("flash-attention submodule list not narrowed", file=sys.stderr)
        return 1

    for stage in ("base", "vllm-runtime-base"):
        if not stage_has(dockerfile, stage, "ENV UV_INDEX_URL"):
            print(f"stage {stage} still has no ENV UV_INDEX_URL", file=sys.stderr)
            return 1

    if "yum.repos.d/cn.repo" not in dockerfile:
        print("dnf still points at the default repositories", file=sys.stderr)
        return 1

    if not stage_has(dockerfile, "rust-build", "ENV RUSTUP_DIST_SERVER"):
        print("rust-build still points at the default Rust servers", file=sys.stderr)
        return 1

    # A doc-comment URL is fine; no index may still resolve there.
    leftover = re.search(
        r"(--index-url|--extra-index-url) https://flashinfer\.ai",
        dockerfile + requirements,
    )
    if leftover:
        print(
            f"an index still points at flashinfer.ai: {leftover.group(0)}",
            file=sys.stderr,
        )
        return 1

    print(f"patched: indexes -> {NEXUS}, rust -> {CARGO_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
