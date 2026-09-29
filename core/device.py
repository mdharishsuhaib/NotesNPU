"""Hardware detection and ONNX Runtime session helpers.

Handles both ways of getting the Qualcomm QNN Execution Provider:
  * onnxruntime-qnn >= 2.0  -> plugin EP registered at runtime
  * onnxruntime-qnn 1.x     -> "QNNExecutionProvider" built into the wheel
and falls back to the CPU EP everywhere else.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from dataclasses import dataclass, field

import onnxruntime as ort

QNN = "QNNExecutionProvider"
CPU = "CPUExecutionProvider"


def is_windows_arm64() -> bool:
    if platform.system() != "Windows":
        return False
    # Native ARM64 Python reports ARM64; x64 Python under emulation still exposes
    # PROCESSOR_ARCHITEW6432 / PROCESSOR_IDENTIFIER hints.
    arch = (platform.machine() or "").upper()
    ident = os.environ.get("PROCESSOR_IDENTIFIER", "").upper()
    return arch == "ARM64" or "ARMV8" in ident or "ARM64" in ident


@functools.lru_cache(maxsize=1)
def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def detect_chipset() -> str | None:
    """Map the CPU name to a Qualcomm AI Hub chipset key used for model download."""
    name = cpu_name().upper()
    if "SNAPDRAGON" not in name and not is_windows_arm64():
        return None
    if "X2" in name:  # X2E-* (X2 Elite), X2P-* (X2 Plus)
        return "x2_elite"
    return "x_elite"  # X1E-* (X Elite), X1P-* (X Plus), X1-* share Hexagon v73


@dataclass
class QnnStatus:
    available: bool = False
    mode: str = "none"  # "plugin" | "builtin" | "none"
    detail: str = ""
    devices: list = field(default_factory=list)


@functools.lru_cache(maxsize=1)
def qnn_status() -> QnnStatus:
    """Detect (and register, if needed) the QNN EP exactly once per process."""
    # 1) Plugin EP (onnxruntime-qnn 2.x)
    try:
        import onnxruntime_qnn as qnn_ep  # type: ignore

        lib = qnn_ep.get_library_path()
        try:
            ort.register_execution_provider_library(QNN, lib)
        except Exception as e:  # already registered is fine
            if "already" not in str(e).lower():
                raise
        devs = [d for d in ort.get_ep_devices() if d.ep_name == QNN]
        if devs:
            return QnnStatus(True, "plugin", f"onnxruntime-qnn {getattr(qnn_ep, '__version__', '?')}", devs)
        return QnnStatus(False, "none", "QNN plugin loaded but no Qualcomm NPU device found")
    except ImportError:
        pass
    except Exception as e:
        return QnnStatus(False, "none", f"QNN plugin error: {e}")

    # 2) Built-in provider bridge (onnxruntime-qnn 1.x)
    if QNN in ort.get_available_providers():
        return QnnStatus(True, "builtin", f"onnxruntime {ort.__version__} (built-in QNN EP)")
    return QnnStatus(False, "none", "QNN EP not installed (not a Snapdragon PC, or onnxruntime-qnn missing)")


def _htp_backend_path() -> str:
    try:
        import onnxruntime_qnn as qnn_ep  # type: ignore

        return qnn_ep.get_qnn_htp_path()
    except Exception:
        return "QnnHtp.dll"


def make_session(model_path: str, prefer_npu: bool = True, threads: int | None = None) -> tuple[ort.InferenceSession, str]:
    """Create an InferenceSession on the NPU if possible, else CPU.

    Returns (session, provider_label).
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3

    st = qnn_status() if prefer_npu else QnnStatus()
    if prefer_npu and st.available:
        qnn_opts = {
            "backend_path": _htp_backend_path(),
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }
        # AI Hub precompiled models are EPContext graphs: they can only run on the NPU,
        # so a QNN failure raises instead of silently falling back to CPU.
        if st.mode == "plugin":
            so.add_provider_for_devices(st.devices, {"backend_path": qnn_opts["backend_path"]})
            sess = ort.InferenceSession(model_path, sess_options=so)
        else:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=[(QNN, qnn_opts), CPU])
        return sess, "NPU (Hexagon, QNN EP)"

    sess = ort.InferenceSession(model_path, sess_options=so, providers=[CPU])
    return sess, "CPU"


def run_options(npu: bool) -> ort.RunOptions | None:
    if not npu:
        return None
    ro = ort.RunOptions()
    try:
        ro.add_run_config_entry("qnn.perf_mode", "burst")
        ro.add_run_config_entry("qnn.rpc_control_latency", "100")
    except Exception:
        pass
    return ro


def system_summary() -> dict:
    st = qnn_status()
    return {
        "cpu": cpu_name(),
        "os": platform.platform(),
        "python": f"{platform.python_version()} ({platform.machine()})",
        "windows_arm64": is_windows_arm64(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "qnn_npu": st.available,
        "qnn_detail": st.detail,
    }
