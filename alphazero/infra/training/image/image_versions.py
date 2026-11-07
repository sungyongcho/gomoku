"""Print the versions a Ray node image must share with every other node image.

Both training images run this at build time. check_image_versions.sh compares the outputs and
fails the build when the GPU head and the CPU workers disagree, because Ray rejects a node whose
Python patch version differs from the head's and mixed library versions break serialization.

Output: one ``name==version`` line per installed distribution, sorted, after ``python``, ``ray``
and ``torch`` lines. Packages only the CUDA build of torch pulls in are left out (the CUDA wheels
and anything required only by them, such as pyelftools for cupti-python), and torch's local tag
(+cpu, +cu134) is stripped, so the CPU and CUDA wheels of one release compare equal.
"""

import platform
import re
from importlib import metadata

from packaging.requirements import Requirement

# Distributions that exist only in the GPU image (CUDA runtime wheels and Triton).
GPU_ONLY = re.compile(r"^(nvidia-.*|cuda-.*|cupti-python|triton)$")
# The project itself is installed from the build context, not from a pinned version, and the
# installer tools do not reach the Ray workers' code paths.
SKIP = {"gmk-alphazero", "pip", "wheel"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _gpu_only(dists: dict[str, metadata.Distribution]) -> set[str]:
    """CUDA wheels plus every package whose only dependents are CUDA wheels."""
    dependents: dict[str, set[str]] = {name: set() for name in dists}
    for name, dist in dists.items():
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            dep = _normalize(req.name)
            if dep in dependents:
                dependents[dep].add(name)

    gpu_only = {name for name in dists if GPU_ONLY.match(name)}
    changed = True
    while changed:
        changed = False
        for name, users in dependents.items():
            if name not in gpu_only and users and users <= gpu_only:
                gpu_only.add(name)
                changed = True
    return gpu_only


def main() -> None:
    dists = {_normalize(d.metadata["Name"]): d for d in metadata.distributions()}
    gpu_only = _gpu_only(dists)
    versions: dict[str, str] = {}
    for name, dist in dists.items():
        if name in SKIP or name in gpu_only:
            continue
        versions[name] = dist.version.split("+", 1)[0] if name == "torch" else dist.version

    print(f"python=={platform.python_version()}")
    print(f"ray=={versions.get('ray', 'missing')}")
    print(f"torch=={versions.get('torch', 'missing')}")
    for name in sorted(versions):
        print(f"{name}=={versions[name]}")


if __name__ == "__main__":
    main()
