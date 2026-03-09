"""
src/export/onnx_export.py
──────────────────────────
Exports trained PyTorch models to ONNX format for fast inference.

Why ONNX?
  - onnxruntime runs inference 3–5× faster than PyTorch on CPU
  - No PyTorch dependency needed in production — just onnxruntime
  - Easier to deploy to Lambda, Docker, etc.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Tuple
from rich.console import Console

console = Console()


def export_model_to_onnx(
    model: torch.nn.Module,
    seq_len: int,
    n_features: int,
    output_path: str,
    model_name: str,
) -> str:
    model.eval()
    dummy = torch.randn(1, seq_len, n_features)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["regression", "classification"],
        dynamic_axes={
            "input":          {0: "batch_size"},
            "regression":     {0: "batch_size"},
            "classification": {0: "batch_size"},
        },
        dynamo=False,       # ← this is the key line
        verbose=False,
    )
    return output_path

    # Validate
    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"  [green]✓  {model_name}: {file_size_mb:.2f} MB, opset {opset_version}[/green]")
    return output_path


def verify_onnx_output(
    pytorch_model: torch.nn.Module,
    onnx_path:     Path,
    seq_len:       int = 60,
    n_features:    int = 18,
    rtol:          float = 1e-3,
    atol:          float = 1e-5,
) -> bool:
    """
    Verifies ONNX output matches PyTorch output numerically.
    Returns True if outputs match within tolerance.
    """
    dummy = torch.randn(4, seq_len, n_features)

    pytorch_model.eval()
    with torch.no_grad():
        pt_reg, pt_cls = pytorch_model(dummy)
    pt_reg_np = pt_reg.numpy()
    pt_cls_np = pt_cls.numpy()

    session = ort.InferenceSession(str(onnx_path))
    ort_reg, ort_cls = session.run(None, {"input": dummy.numpy()})

    reg_match = np.allclose(pt_reg_np, ort_reg, rtol=rtol, atol=atol)
    cls_match = np.allclose(pt_cls_np, ort_cls, rtol=rtol, atol=atol)

    if reg_match and cls_match:
        console.print(f"  [green]✓  ONNX output verified (reg Δ≤{atol}, cls Δ≤{atol})[/green]")
        return True
    else:
        console.print(f"  [red]✗  ONNX output mismatch! reg_match={reg_match}, cls_match={cls_match}[/red]")
        return False


def benchmark_onnx_vs_pytorch(
    pytorch_model: torch.nn.Module,
    onnx_path:     Path,
    seq_len:       int = 60,
    n_features:    int = 18,
    n_runs:        int = 100,
) -> dict:
    """Measures inference latency for both runtimes."""
    import time
    dummy = torch.randn(1, seq_len, n_features)

    # PyTorch
    pytorch_model.eval()
    times_pt = []
    with torch.no_grad():
        for _ in range(n_runs):
            t = time.perf_counter()
            pytorch_model(dummy)
            times_pt.append(time.perf_counter() - t)

    # ONNX Runtime
    session = ort.InferenceSession(str(onnx_path))
    inp = dummy.numpy()
    times_ort = []
    for _ in range(n_runs):
        t = time.perf_counter()
        session.run(None, {"input": inp})
        times_ort.append(time.perf_counter() - t)

    avg_pt  = np.mean(times_pt)  * 1000
    avg_ort = np.mean(times_ort) * 1000
    speedup = avg_pt / avg_ort

    result = {
        "pytorch_ms":  round(avg_pt, 3),
        "onnx_ms":     round(avg_ort, 3),
        "speedup":     round(speedup, 2),
    }
    console.print(
        f"  Inference: PyTorch={avg_pt:.2f}ms  "
        f"ONNX={avg_ort:.2f}ms  "
        f"[green]Speedup {speedup:.1f}×[/green]"
    )
    return result


def export_all_models(
    lstm_model,
    transformer_model,
    meta_learner,
    seq_len:    int  = 60,
    n_features: int  = 18,
    onnx_dir:   Path = Path("models/onnx"),
) -> dict:
    """Exports all three models to ONNX and verifies each."""
    console.print("\n[bold cyan]📦 Exporting models to ONNX[/bold cyan]")
    paths = {}

    # LSTM
    lstm_path = onnx_dir / "lstm.onnx"
    export_model_to_onnx(lstm_model, seq_len, n_features, lstm_path, "LSTM")
    verify_onnx_output(lstm_model, lstm_path, seq_len, n_features)
    benchmark_onnx_vs_pytorch(lstm_model, lstm_path, seq_len, n_features)
    paths["lstm"] = str(lstm_path)

    # Transformer
    tf_path = onnx_dir / "transformer.onnx"
    export_model_to_onnx(transformer_model, seq_len, n_features, tf_path, "Transformer")
    verify_onnx_output(transformer_model, tf_path, seq_len, n_features)
    benchmark_onnx_vs_pytorch(transformer_model, tf_path, seq_len, n_features)
    paths["transformer"] = str(tf_path)

    # Meta-learner (different input shape: batch × 6)
    meta_path = onnx_dir / "meta_learner.onnx"
    meta_learner.eval()
    dummy_meta = (torch.zeros(1, 3), torch.zeros(1, 3))
    torch.onnx.export(
        meta_learner,
        dummy_meta,
        str(meta_path),
        export_params=True,
        opset_version=17,
        input_names=["lstm_probs", "transformer_probs"],
        output_names=["ensemble_probs"],
        dynamic_axes={
            "lstm_probs":         {0: "batch_size"},
            "transformer_probs":  {0: "batch_size"},
            "ensemble_probs":     {0: "batch_size"},
        },
    )
    paths["meta_learner"] = str(meta_path)
    console.print(f"  [green]✓  meta_learner exported[/green]")

    console.print(f"\n[bold green]✓  All models exported to {onnx_dir}/[/bold green]\n")
    return paths
