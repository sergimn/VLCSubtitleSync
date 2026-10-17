"""Can Whisper use an NVIDIA GPU here? Checked by ``vlc-subsync setup`` and ``doctor``.

faster-whisper runs on ctranslate2, which loads the CUDA libraries it needs (cuBLAS)
at run time and does not ship them. Without them ``device=auto`` quietly falls back
to the CPU (only the helper log says so). ``ctranslate2.get_cuda_device_count()``
only asks the driver, so it reports a GPU even when cuBLAS is missing: we also try to
load the library files.

Which files: those the installed ctranslate2 binary names (read from the binary, so
the check follows ctranslate2 upgrades). 4.8.3 names only ``libcublas.so.12``
(``cublas64_12.dll``) besides the driver; it does not use cuDNN.

The check only reports. Nothing is installed: the user is told which packages provide
the missing libraries for the package manager we find.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import mmap
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .setup_vlc import Context

# CUDA library names as they appear in ctranslate2's binaries. The driver's own
# library (libcuda.so.1, nvcuda.dll) is left out: the device count covers it.
_LIBRARY_RE = {
    "linux": re.compile(rb"lib(?:cu|nvrtc)[A-Za-z_]*\.so\.[0-9]+"),
    "windows": re.compile(rb"(?:cu|nvrtc)[A-Za-z_]*64_[0-9]+\.dll", re.IGNORECASE),
}
_DRIVER_LIBRARIES = {"libcuda.so.1"}
# Used when the binaries cannot be read: what ctranslate2 4.x needs.
FALLBACK_LIBRARIES: dict[str, tuple[str, ...]] = {
    "linux": ("libcublas.so.12",),
    "windows": ("cublas64_12.dll",),
}
_CUBLAS12 = {"libcublas.so.12", "libcublasLt.so.12", "cublas64_12.dll", "cublasLt64_12.dll"}

# Linux package managers in the order we look for them, with the packages providing
# cuBLAS for CUDA 12 (checked 2026-10). CUDA 13 packages do not work: the file names
# carry the CUDA major version.
PACKAGE_HINTS: dict[str, tuple[str, ...]] = {
    "apt-get": (
        "Ubuntu 24.04+ / Debian 13+: sudo apt install libcublas12",
        "  (Ubuntu multiverse, Debian non-free). Older releases: libcublas-12-9",
        "  from NVIDIA's CUDA repository (https://developer.nvidia.com/cuda-downloads)",
    ),
    "dnf": (
        "Fedora/RHEL: libcublas-12-9 from NVIDIA's CUDA repository",
        "  (https://developer.nvidia.com/cuda-downloads; its RHEL 9 and Fedora 41",
        "  repositories have it, newer Fedora ones only have CUDA 13)",
    ),
    "zypper": (
        "openSUSE/SLES: libcublas-12-9 from NVIDIA's CUDA repository",
        "  (https://developer.nvidia.com/cuda-downloads)",
    ),
    "pacman": (
        "Arch: the `cuda` package is CUDA 13, which does not work; install a CUDA 12",
        "  build instead, for example the AUR package cuda12.0",
    ),
}

WINDOWS_HINT = (
    "Windows: the CUDA Toolkit 12.x from NVIDIA (it includes cuBLAS,",
    "  https://developer.nvidia.com/cuda-toolkit-archive), with its bin folder on PATH",
)

DRIVER_HINT = ("Update the NVIDIA driver: CUDA 12 needs driver 525 or newer (Windows: 527).",)


def _scan(binaries: Sequence[Path], platform: str) -> tuple[str, ...]:
    """CUDA library names found in ``binaries`` (sorted, driver library excluded)."""
    pattern = _LIBRARY_RE[platform]
    names: set[str] = set()
    for path in binaries:
        try:
            with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                names.update(x.decode("ascii") for x in pattern.findall(m))
        except (OSError, ValueError):
            continue
    if platform == "windows":
        # case varies in the binary; LoadLibrary does not care, the hints do
        canon = {n.lower(): n for n in FALLBACK_LIBRARIES["windows"] + ("cublasLt64_12.dll",)}
        names = {canon.get(n.lower(), n) for n in names}
    return tuple(sorted(names - _DRIVER_LIBRARIES))


def _ctranslate2_binaries(platform: str) -> list[Path]:
    """ctranslate2's native libraries, located without importing it."""
    try:
        spec = importlib.util.find_spec("ctranslate2")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    pkg = Path(next(iter(spec.submodule_search_locations)))
    if platform == "windows":
        return sorted(pkg.glob("*.dll")) + sorted(pkg.glob("*.pyd"))
    # auditwheel puts the bundled library next to the package
    return sorted(pkg.parent.glob("ctranslate2.libs/libctranslate2*.so*")) + sorted(
        pkg.glob("*.so")
    )


@functools.cache
def required_libraries(platform: str) -> tuple[str, ...]:
    """CUDA library files the installed ctranslate2 loads at run time."""
    if platform not in _LIBRARY_RE:
        return ()
    return _scan(_ctranslate2_binaries(platform), platform) or FALLBACK_LIBRARIES[platform]


def install_hint(ctx: Context, missing: Sequence[str]) -> list[str]:
    """Lines telling the user which packages provide the ``missing`` libraries."""
    if not set(missing) <= _CUBLAS12:
        # a newer ctranslate2 needing something we have no package names for
        lines = ["install the NVIDIA CUDA libraries that provide " + ", ".join(missing)]
    elif ctx.platform == "windows":
        lines = list(WINDOWS_HINT)
    else:
        pm = next((p for p in PACKAGE_HINTS if ctx.which(p)), None)
        lines = list(PACKAGE_HINTS[pm]) if pm else ["install cuBLAS for CUDA 12"]
        lines.append("CUDA 13 versions do not work: SubSync needs the CUDA 12 ones.")
    lines.append("Install them system-wide: the helper VLC starts does not see a")
    lines.append("LD_LIBRARY_PATH or PATH set only in your shell.")
    lines.append("SubSync works without them, on the CPU (slower). After installing")
    lines.append("them, restart VLC; `vlc-subsync doctor` shows whether they are found.")
    return lines


def nvidia_gpu(ctx: Context) -> str | None:
    """Name of the first NVIDIA GPU that ``nvidia-smi -L`` lists, or None.

    None also when nvidia-smi is missing (some distributions package it apart from
    the driver), fails or times out: the device count decides then.
    """
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


def cuda_device_count() -> int:
    """GPUs the driver offers to CUDA (0 also when the driver is too old for CUDA 12)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        return 0


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
    driver_ok: bool = True  # False: a GPU, but CUDA sees no device (driver too old?)
    missing: list[str] = field(default_factory=list)  # CUDA library files not found

    @property
    def usable(self) -> bool:
        return self.gpu is not None and self.driver_ok and not self.missing


def gpu_status(
    ctx: Context,
    load: Callable[[str, str], bool] | None = None,
    count: Callable[[], int] | None = None,
    libraries: Sequence[str] | None = None,
) -> GpuStatus:
    """Is there an NVIDIA GPU, does CUDA see it, and which CUDA libraries are missing?

    ``load``/``count`` default to :func:`can_load`/:func:`cuda_device_count`, looked
    up at call time so tests can replace them.
    """
    if ctx.platform == "macos":
        return GpuStatus(None)
    load = load or can_load
    name = nvidia_gpu(ctx)
    devices = (count or cuda_device_count)()
    if name is None and devices == 0:
        return GpuStatus(None)
    st = GpuStatus(name or "NVIDIA GPU", driver_ok=devices > 0)
    libs = required_libraries(ctx.platform) if libraries is None else libraries
    st.missing = [n for n in libs if not load(n, ctx.platform)]
    return st


def report(ctx: Context, **probe) -> GpuStatus:
    """Setup step: say whether Whisper will use the GPU, and what to install if not.

    ``probe`` is passed to :func:`gpu_status` (tests replace the system probes).
    """
    ctx.step("GPU")
    if ctx.platform == "macos":
        ctx.info("no CUDA on macOS: Whisper runs on the CPU")
        return GpuStatus(None)
    st = gpu_status(ctx, **probe)
    if st.gpu is None:
        ctx.info("no NVIDIA GPU found: Whisper runs on the CPU")
    elif st.usable:
        ctx.ok(f"{st.gpu}: CUDA libraries found, Whisper will use the GPU")
    elif not st.driver_ok:
        ctx.warn(f"{st.gpu} found, but CUDA cannot use it: Whisper runs on the CPU")
        for line in DRIVER_HINT:
            ctx.info(line)
    else:
        ctx.warn(f"{st.gpu} found, but Whisper cannot use it: missing {', '.join(st.missing)}")
        for line in install_hint(ctx, st.missing):
            ctx.info(line)
    return st
