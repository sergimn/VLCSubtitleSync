"""Can Whisper use an NVIDIA GPU here? Checked by ``vlc-subsync setup`` and ``doctor``.

faster-whisper runs on ctranslate2, which loads cuBLAS for CUDA 12 and cuDNN 9 at run
time and does not ship them. Without them ``device=auto`` quietly falls back to the CPU
(only the helper log says so). ``ctranslate2.get_cuda_device_count()`` counts devices
through the driver alone, so it reports a GPU even when these libraries are missing:
we try to load the library files instead.

The check only reports. Nothing is installed: the user is told which packages provide
the missing libraries for the package manager we find.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .setup_vlc import Context

# The files ctranslate2 4.x loads. Sonames are per CUDA major version: CUDA 13's
# libcublas.so.13 does not satisfy a request for libcublas.so.12.
CUDA_LIBRARIES: dict[str, tuple[str, ...]] = {
    "linux": ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"),
    "windows": ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"),
}

LIBRARY_NAMES = "cuBLAS for CUDA 12 and cuDNN 9 for CUDA 12"

# Linux package managers in the order we look for them, with the packages that
# provide the libraries above. CUDA 13 packages do not work, which rules out e.g.
# Arch's `cuda` package and NVIDIA's newer Fedora repositories.
PACKAGE_HINTS: dict[str, tuple[str, ...]] = {
    "apt-get": (
        "Debian/Ubuntu: sudo apt install libcublas12 libcublaslt12",
        "  (Ubuntu multiverse, Debian non-free), and libcudnn9-cuda-12 from",
        "  NVIDIA's CUDA repository (https://developer.nvidia.com/cudnn-downloads)",
    ),
    "dnf": (
        "Fedora/RHEL: libcublas-12-9 and libcudnn9-cuda-12 from NVIDIA's CUDA",
        "  repository (https://developer.nvidia.com/cuda-downloads)",
    ),
    "zypper": (
        "openSUSE/SLES: libcublas-12-9 and libcudnn9-cuda-12 from NVIDIA's CUDA",
        "  repository (https://developer.nvidia.com/cuda-downloads)",
    ),
    "pacman": (
        "Arch: the `cuda` package is CUDA 13, which does not work; install CUDA 12",
        "  builds of cuBLAS and cuDNN 9 (for example from the AUR)",
    ),
}

WINDOWS_HINT = (
    "Windows: from NVIDIA, the CUDA Toolkit 12.x (it includes cuBLAS) and cuDNN 9",
    "  for CUDA 12; make sure their bin folders are on PATH",
)


def install_hint(ctx: Context) -> list[str]:
    """Lines telling the user which packages provide the CUDA libraries."""
    if ctx.platform == "windows":
        lines = list(WINDOWS_HINT)
    else:
        pm = next((p for p in PACKAGE_HINTS if ctx.which(p)), None)
        lines = list(PACKAGE_HINTS[pm]) if pm else [f"install {LIBRARY_NAMES}"]
    lines.append("CUDA 13 versions do not work: SubSync needs the CUDA 12 ones.")
    lines.append("SubSync works without them, on the CPU (slower). After installing")
    lines.append("them, restart VLC; `vlc-subsync doctor` shows whether they are found.")
    return lines


def nvidia_gpu(ctx: Context) -> str | None:
    """Name of the first NVIDIA GPU that ``nvidia-smi -L`` lists, or None."""
    if ctx.platform == "macos":
        return None
    exe = ctx.which("nvidia-smi")
    if not exe:
        return None
    r = ctx.sh([exe, "-L"], timeout=15)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        # "GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU (UUID: GPU-...)"
        if line.startswith("GPU ") and ":" in line:
            name = line.split(":", 1)[1].split("(UUID", 1)[0].strip()
            return name or "NVIDIA GPU"
    return None


def _prepare_windows_search() -> None:
    # ctranslate2's __init__ registers its package folder and loads the DLLs bundled
    # there; a library already loaded under that name is found by LoadLibrary below.
    try:
        import ctranslate2  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def can_load(name: str, platform: str) -> bool:
    """True if the dynamic loader finds library ``name`` the way ctranslate2 would."""
    try:
        if platform == "windows":
            _prepare_windows_search()
            ctypes.CDLL(name, winmode=0)  # winmode=0: the classic search, PATH included
        else:
            ctypes.CDLL(name)
    except OSError:
        return False
    return True


@dataclass
class GpuStatus:
    gpu: str | None  # NVIDIA GPU name, None when there is none (or on macOS)
    missing: list[str] = field(default_factory=list)  # CUDA library files not found

    @property
    def usable(self) -> bool:
        return self.gpu is not None and not self.missing


def gpu_status(ctx: Context, load: Callable[[str, str], bool] = can_load) -> GpuStatus:
    gpu = nvidia_gpu(ctx)
    if gpu is None:
        return GpuStatus(None)
    libs = CUDA_LIBRARIES.get(ctx.platform, ())
    return GpuStatus(gpu, [n for n in libs if not load(n, ctx.platform)])


def report(ctx: Context, load: Callable[[str, str], bool] = can_load) -> GpuStatus:
    """Setup step: say whether Whisper will use the GPU, and what to install if not."""
    ctx.step("GPU")
    if ctx.platform == "macos":
        ctx.info("no CUDA on macOS: Whisper runs on the CPU")
        return GpuStatus(None)
    st = gpu_status(ctx, load)
    if st.gpu is None:
        ctx.info("no NVIDIA GPU found: Whisper runs on the CPU")
    elif st.usable:
        ctx.ok(f"{st.gpu}: CUDA libraries found, Whisper will use the GPU")
    else:
        ctx.warn(f"{st.gpu} found, but Whisper cannot use it: missing {', '.join(st.missing)}")
        for line in install_hint(ctx):
            ctx.info(line)
    return st
