import subprocess
import sys

import pytest

import vlcsubsync.gpu as G
import vlcsubsync.setup_vlc as S
from vlcsubsync import cli

SMI_OUT = "GPU 0: NVIDIA GeForce RTX 3050 Ti Laptop GPU (UUID: GPU-1234)\n"


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


def loader(missing=()):
    tried: list[str] = []

    def load(name, platform):
        tried.append(name)
        return name not in missing

    load.tried = tried
    return load


def never_load(name, platform):
    raise AssertionError(f"tried to load {name} without a GPU")


# ------------------------------------------------------------------ GPU detection


def test_nvidia_gpu_name_from_nvidia_smi(tmp_path):
    ctx, _, calls = make_ctx(tmp_path)
    assert G.nvidia_gpu(ctx) == "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
    assert calls == [["/usr/bin/nvidia-smi", "-L"]]


@pytest.mark.parametrize(
    "kw",
    [
        {"tools": ()},  # no nvidia-smi: no NVIDIA driver
        {"smi_rc": 9},  # driver not loaded
        {"smi_out": "No devices were found\n"},
    ],
)
def test_no_nvidia_gpu(tmp_path, kw):
    ctx, _, _ = make_ctx(tmp_path, **kw)
    assert G.nvidia_gpu(ctx) is None


def test_macos_never_runs_nvidia_smi(tmp_path):
    ctx, _, calls = make_ctx(tmp_path, platform="macos")
    assert G.nvidia_gpu(ctx) is None
    assert calls == []


# ------------------------------------------------------------------ library check


def test_status_lists_missing_libraries(tmp_path):
    ctx, _, _ = make_ctx(tmp_path)
    load = loader(missing={"libcublas.so.12", "libcudnn.so.9"})
    st = G.gpu_status(ctx, load)
    assert load.tried == list(G.CUDA_LIBRARIES["linux"])
    assert st.missing == ["libcublas.so.12", "libcudnn.so.9"]
    assert not st.usable


def test_status_checks_windows_dll_names(tmp_path):
    ctx, _, _ = make_ctx(tmp_path, platform="windows")
    load = loader()
    assert G.gpu_status(ctx, load).usable
    assert load.tried == ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"]


def test_status_without_gpu_loads_nothing(tmp_path):
    ctx, _, _ = make_ctx(tmp_path, tools=())
    st = G.gpu_status(ctx, never_load)
    assert st.gpu is None and st.missing == [] and not st.usable


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sonames")
def test_can_load_real_loader():
    assert G.can_load("libc.so.6", "linux")
    assert not G.can_load("libvlcsubsync-does-not-exist.so.12", "linux")


# ------------------------------------------------------------------ setup report


def test_report_gpu_usable(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path)
    G.report(ctx, loader())
    assert lines[-1] == (
        "  [ok]   NVIDIA GeForce RTX 3050 Ti Laptop GPU: CUDA libraries found, "
        "Whisper will use the GPU"
    )
    assert ctx.warnings == []


def test_report_missing_libraries_names_apt_packages(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, tools=("nvidia-smi", "apt-get", "dnf"))
    G.report(ctx, loader(missing={"libcublas.so.12", "libcublasLt.so.12"}))
    text = "\n".join(lines)
    assert ctx.warnings == [
        "NVIDIA GeForce RTX 3050 Ti Laptop GPU found, but Whisper cannot use it: "
        "missing libcublas.so.12, libcublasLt.so.12"
    ]
    assert "sudo apt install libcublas12 libcublaslt12" in text
    assert "libcudnn9-cuda-12" in text
    assert "Fedora" not in text  # only the first package manager found
    assert "CUDA 13 versions do not work" in text
    assert "on the CPU" in text


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("dnf", "libcublas-12-9 and libcudnn9-cuda-12"),
        ("zypper", "openSUSE/SLES: libcublas-12-9"),
        ("pacman", "the `cuda` package is CUDA 13"),
        (None, "install cuBLAS for CUDA 12 and cuDNN 9 for CUDA 12"),
    ],
)
def test_report_hint_per_package_manager(tmp_path, tool, expected):
    tools = ("nvidia-smi", tool) if tool else ("nvidia-smi",)
    ctx, lines, _ = make_ctx(tmp_path, tools=tools)
    G.report(ctx, loader(missing={"libcudnn.so.9"}))
    assert expected in "\n".join(lines)


def test_report_windows_hint(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, platform="windows")
    G.report(ctx, loader(missing={"cudnn64_9.dll"}))
    text = "\n".join(lines)
    assert "missing cudnn64_9.dll" in text
    assert "CUDA Toolkit 12.x" in text and "on PATH" in text


def test_report_no_gpu_is_quiet(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, tools=("apt-get",))
    G.report(ctx, never_load)
    assert lines[-1].strip() == "no NVIDIA GPU found: Whisper runs on the CPU"
    assert ctx.warnings == []


def test_report_macos(tmp_path):
    ctx, lines, _ = make_ctx(tmp_path, platform="macos")
    G.report(ctx, never_load)
    assert lines[-1].strip() == "no CUDA on macOS: Whisper runs on the CPU"


# ------------------------------------------------------------------ setup / doctor


def test_setup_ends_with_gpu_notice(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG", "CONFIG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / "app" / name.lower()))
    missing = G.GpuStatus("Test GPU", ["libcublas.so.12"])
    monkeypatch.setattr(G, "gpu_status", lambda ctx, load=None: missing)
    ctx, lines, _ = make_ctx(tmp_path / "sys", tools=("apt-get",))
    (tmp_path / "sys").mkdir()
    assert S.run_setup(ctx, model=False, autostart=False) == 0  # a warning, not an error
    text = "\n".join(lines)
    gpu = text.index("\nGPU\n")
    assert text.index("Test GPU found, but Whisper cannot use it", gpu) < text.index("Done!")
    assert "libcublas12" in text[gpu:]


def test_setup_survives_a_failing_gpu_check(tmp_path, monkeypatch):
    for name in ("CACHE", "STATE", "LOG", "CONFIG"):
        monkeypatch.setenv(f"VLC_SUBSYNC_{name}_DIR", str(tmp_path / "app" / name.lower()))

    def boom(ctx, load=None):
        raise RuntimeError("nvidia-smi hung")

    monkeypatch.setattr(G, "gpu_status", boom)
    (tmp_path / "sys").mkdir()
    ctx, lines, _ = make_ctx(tmp_path / "sys")
    assert S.run_setup(ctx, model=False, autostart=False) == 0
    assert "GPU check skipped (nvidia-smi hung)" in "\n".join(lines)


def test_doctor_names_missing_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(
        G, "gpu_status", lambda ctx, load=None: G.GpuStatus("Test GPU", ["libcudnn.so.9"])
    )
    (tmp_path / "home").mkdir()
    ctx = S.Context(
        platform="linux",
        home=tmp_path / "home",
        env={},
        root=tmp_path / "root",
        which=lambda n: "/usr/bin/pacman" if n == "pacman" else None,
    )
    lines: list[str] = []
    cli.run_doctor(out=lines.append, ctx=ctx)
    text = "\n".join(lines)
    assert "CUDA: Test GPU, but missing libcudnn.so.9 (CPU will be used)" in text
    assert "the `cuda` package is CUDA 13" in text
