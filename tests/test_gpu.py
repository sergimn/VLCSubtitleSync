import importlib.util
import re
import subprocess
import sys

import pytest

import vlcsubsync.gpu as G
import vlcsubsync.setup_vlc as S
from vlcsubsync import cli

SMI_OUT = "GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU (UUID: GPU-1234)\n"
GPU = "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
CUBLAS = ("libcublas.so.12",)


def make_ctx(tmp_path, platform="linux", *, tools=("nvidia-smi",), smi_rc=0, smi_out=SMI_OUT):
    lines: list[str] = []
    calls: list[list[str]] = []

    def run(args, **kw):
        calls.append(list(args))
        if args[0].endswith("nvidia-smi"):
            return subprocess.CompletedProcess(args, smi_rc, smi_out, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    ctx = S.Context(
        platform=platform,
        home=tmp_path,
        env={},
        root=tmp_path,
        run=run,
        which=lambda n: f"/usr/bin/{n}" if n in tools else None,
        out=lines.append,
        vlc_running=lambda: False,
    )
    return ctx, lines, calls


def probe(missing=(), devices=1, libraries=CUBLAS):
    """gpu_status/report keyword arguments replacing the system probes."""

    return {
        "load": lambda name, platform: name not in missing,
        "count": lambda: devices,
        "libraries": libraries,
    }


def no_gpu_probe():
    def load(name, platform):
        raise AssertionError(f"tried to load {name} without a GPU")

    return {"load": load, "count": lambda: 0, "libraries": CUBLAS}


@pytest.fixture
def app_dirs(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG", "CONFIG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / "app" / name.lower()))


# ------------------------------------------------------------------ GPU detection


def test_nvidia_gpu_name_from_nvidia_smi(tmp_path):
    ctx, _, calls = make_ctx(tmp_path)
    assert G.nvidia_gpu(ctx) == GPU
    assert calls == [["/usr/bin/nvidia-smi", "-L"]]


@pytest.mark.parametrize(
    "kw",
    [
        {"tools": ()},  # no nvidia-smi
        {"smi_rc": 9},  # driver not loaded
        {"smi_rc": 127},  # what Context.sh returns for a timeout
        {"smi_out": "No devices were found\n"},
    ],
)
def test_nvidia_gpu_none(tmp_path, kw):
    ctx, _, _ = make_ctx(tmp_path, **kw)
    assert G.nvidia_gpu(ctx) is None


def test_macos_never_probes(tmp_path):
    ctx, _, calls = make_ctx(tmp_path, platform="macos")
    assert G.nvidia_gpu(ctx) is None
    assert G.gpu_status(ctx, **no_gpu_probe()) == G.GpuStatus(None)
    assert calls == []


# ------------------------------------------------------------------ which libraries


def test_scan_finds_cuda_libraries_but_not_the_driver(tmp_path):
    lib = tmp_path / "libctranslate2.so.4"
    lib.write_bytes(b"\x7fELF\0libcublas.so.12\0libcuda.so.1\0libgomp.so.1\0libcudnn.so.9\0")
    assert G._scan([lib, tmp_path / "missing.so"], "linux") == ("libcublas.so.12", "libcudnn.so.9")


def test_scan_windows_names(tmp_path):
    dll = tmp_path / "ctranslate2.dll"
    dll.write_bytes(b"MZ\0CUBLAS64_12.dll\0nvcuda.dll\0cudnn64_9.dll\0KERNEL32.dll\0")
    assert G._scan([dll], "windows") == ("cublas64_12.dll", "cudnn64_9.dll")


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or importlib.util.find_spec("ctranslate2") is None,
    reason="needs ctranslate2 on Linux",
)
def test_required_libraries_from_installed_ctranslate2():
    libs = G.required_libraries("linux")
    assert libs and all(re.fullmatch(r"lib\w+\.so\.\d+", n) for n in libs)
    assert "libcuda.so.1" not in libs  # the driver: covered by the device count


def test_required_libraries_unknown_platform():
    assert G.required_libraries("macos") == ()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sonames")
def test_can_load_real_loader():
    assert G.can_load("libc.so.6", "linux")
    assert not G.can_load("libvlcsubsync-does-not-exist.so.12", "linux")


# ------------------------------------------------------------------ status


def test_status_usable(tmp_path):
    ctx, _, _ = make_ctx(tmp_path)
    st = G.gpu_status(ctx, **probe())
    assert st == G.GpuStatus(GPU) and st.usable


def test_status_lists_missing_libraries(tmp_path):
    ctx, _, _ = make_ctx(tmp_path)
    st = G.gpu_status(ctx, **probe(missing={"libcublas.so.12"}, libraries=CUBLAS + ("x.so.1",)))
    assert st.missing == ["libcublas.so.12"] and not st.usable


def test_status_driver_without_cuda_device(tmp_path):
    # nvidia-smi sees the GPU, CUDA does not (driver too old for CUDA 12)
    ctx, _, _ = make_ctx(tmp_path)
    st = G.gpu_status(ctx, **probe(devices=0))
    assert st.gpu == GPU and not st.driver_ok and not st.usable


@pytest.mark.parametrize("kw", [{"tools": ()}, {"smi_rc": 127}])
def test_status_falls_back_to_device_count(tmp_path, kw):
    # nvidia-smi packaged separately (missing) or timing out: CUDA's count decides
    ctx, _, _ = make_ctx(tmp_path, **kw)
    st = G.gpu_status(ctx, **probe())
    assert st.gpu == "NVIDIA GPU" and st.usable


def test_status_without_gpu_loads_nothing(tmp_path):
    ctx, _, _ = make_ctx(tmp_path, tools=())
    assert G.gpu_status(ctx, **no_gpu_probe()) == G.GpuStatus(None)


# ------------------------------------------------------------------ setup report


def test_report_gpu_usable(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path)
    G.report(ctx, **probe())
    assert lines[-1] == f"  [ok]   {GPU}: CUDA libraries found, Whisper will use the GPU"
    assert ctx.warnings == []


def test_report_missing_cublas_names_apt_packages(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, tools=("nvidia-smi", "apt-get", "dnf"))
    G.report(ctx, **probe(missing={"libcublas.so.12"}))
    text = "\n".join(lines)
    assert ctx.warnings == [f"{GPU} found, but Whisper cannot use it: missing libcublas.so.12"]
    assert "Ubuntu 24.04+ / Debian 13+: sudo apt install libcublas12" in text
    assert "Older releases: libcublas-12-9" in text
    assert "cudnn" not in text.lower()  # ctranslate2 4.x does not use cuDNN
    assert "Fedora" not in text  # only the first package manager found
    assert "CUDA 13 versions do not work" in text
    assert "system-wide" in text and "on the CPU" in text


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("dnf", "Fedora/RHEL: libcublas-12-9"),
        ("zypper", "openSUSE/SLES: libcublas-12-9"),
        ("pacman", "the AUR package cuda12.0"),
        (None, "install cuBLAS for CUDA 12"),
    ],
)
def test_report_hint_per_package_manager(tmp_path, tool, expected):
    tools = ("nvidia-smi", tool) if tool else ("nvidia-smi",)
    ctx, lines, _ = make_ctx(tmp_path, tools=tools)
    G.report(ctx, **probe(missing={"libcublas.so.12"}))
    assert expected in "\n".join(lines)


def test_report_windows_hint(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, platform="windows")
    G.report(ctx, **probe(missing={"cublas64_12.dll"}, libraries=("cublas64_12.dll",)))
    text = "\n".join(lines)
    assert "missing cublas64_12.dll" in text
    assert "CUDA Toolkit 12.x" in text and "on PATH" in text


def test_report_unknown_library_gets_a_generic_hint(tmp_path):
    # a future ctranslate2 needing a library we have no package names for
    ctx, lines, _ = make_ctx(tmp_path, tools=("nvidia-smi", "apt-get"))
    G.report(ctx, **probe(missing={"libcudnn.so.9"}, libraries=("libcudnn.so.9",)))
    text = "\n".join(lines)
    assert "install the NVIDIA CUDA libraries that provide libcudnn.so.9" in text
    assert "apt install" not in text


def test_report_driver_hint(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, tools=("nvidia-smi", "apt-get"))
    G.report(ctx, **probe(devices=0))
    text = "\n".join(lines)
    assert ctx.warnings == [f"{GPU} found, but CUDA cannot use it: Whisper runs on the CPU"]
    assert "Update the NVIDIA driver" in text and "apt install" not in text


def test_report_no_gpu_is_quiet(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, tools=("apt-get",))
    G.report(ctx, **no_gpu_probe())
    assert lines[-1].strip() == "no NVIDIA GPU found: Whisper runs on the CPU"
    assert ctx.warnings == []


def test_report_macos(tmp_path):
    ctx, lines, calls = make_ctx(tmp_path, platform="macos")
    G.report(ctx, **no_gpu_probe())
    assert lines[-1].strip() == "no CUDA on macOS: Whisper runs on the CPU"
    assert calls == []


# ------------------------------------------------------------------ setup


def setup_lines(tmp_path, monkeypatch, status):
    if isinstance(status, Exception):

        def fake(ctx, **kw):
            raise status

    else:

        def fake(ctx, **kw):
            return status

    monkeypatch.setattr(G, "gpu_status", fake)
    (tmp_path / "sys").mkdir()
    ctx, lines, _ = make_ctx(tmp_path / "sys", tools=("apt-get",))
    assert S.run_setup(ctx, model=False, autostart=False) == 0  # never an error
    return "\n".join(lines)


def test_setup_ends_with_gpu_notice(tmp_path, monkeypatch, app_dirs):
    text = setup_lines(tmp_path, monkeypatch, G.GpuStatus("Test GPU", missing=["libcublas.so.12"]))
    gpu = text.index("\nGPU\n")
    done = text.index("Done!")
    assert gpu < text.index("Test GPU found, but Whisper cannot use it") < done
    assert "sudo apt install libcublas12" in text[gpu:done]
    assert text.endswith(
        "Whisper will run on the CPU (slower): see 'GPU' above for what to install."
    )


def test_setup_without_gpu_has_no_cpu_note(tmp_path, monkeypatch, app_dirs):
    text = setup_lines(tmp_path, monkeypatch, G.GpuStatus(None))
    assert "no NVIDIA GPU found" in text
    assert "Whisper will run on the CPU (slower)" not in text


def test_setup_survives_an_unexpected_gpu_check_error(tmp_path, monkeypatch, app_dirs):
    text = setup_lines(tmp_path, monkeypatch, RuntimeError("boom"))
    assert "GPU check skipped (boom)" in text
    assert "Whisper will run on the CPU (slower)" not in text


# ------------------------------------------------------------------ doctor


def doctor_lines(tmp_path, monkeypatch, status, platform="linux", tools=("pacman",)):
    monkeypatch.setattr(G, "gpu_status", lambda ctx, **kw: status)
    (tmp_path / "home").mkdir()
    ctx = S.Context(
        platform=platform,
        home=tmp_path / "home",
        env={},
        root=tmp_path / "root",
        which=lambda n: f"/usr/bin/{n}" if n in tools else None,
    )
    lines: list[str] = []
    cli.run_doctor(out=lines.append, ctx=ctx)
    return lines


def test_doctor_names_missing_libraries(tmp_path, monkeypatch):
    lines = doctor_lines(
        tmp_path, monkeypatch, G.GpuStatus("Test GPU", missing=["libcublas.so.12"])
    )
    i = lines.index("        CUDA: Test GPU, but missing libcublas.so.12 (CPU will be used)")
    assert (
        lines[i + 1]
        == "          Arch: the `cuda` package is CUDA 13, which does not work; install a CUDA 12"
    )


@pytest.mark.parametrize(
    ("status", "platform", "expected"),
    [
        (
            G.GpuStatus("Test GPU"),
            "linux",
            "  [ok]  CUDA: Test GPU: libraries found (GPU will be used)",
        ),
        (G.GpuStatus(None), "linux", "        CUDA: no NVIDIA GPU found (CPU will be used)"),
        (
            G.GpuStatus("Test GPU", driver_ok=False),
            "linux",
            "        CUDA: Test GPU, but CUDA sees no device (CPU will be used)",
        ),
        (None, "macos", "        CUDA: not available on macOS (CPU will be used)"),
    ],
)
def test_doctor_cuda_line(tmp_path, monkeypatch, status, platform, expected):
    lines = doctor_lines(tmp_path, monkeypatch, status, platform=platform)
    assert expected in lines
    if status is not None and not status.driver_ok:
        assert lines[lines.index(expected) + 1].strip().startswith("Update the NVIDIA driver")
