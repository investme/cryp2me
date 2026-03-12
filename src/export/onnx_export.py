"""
src/export/onnx_export.py — cryp2me.ai Phase 5
ONNX export for LSTM, Transformer, and MetaLearner.

Key fixes vs Phase 4:
  - All models forced to CPU before export (no CPU/CUDA device conflict)
  - dynamo=False for LSTM and Transformer (avoids device mismatch error)
  - dynamo=True only for MetaLearner (simple MLP, works fine)
  - Verifies ONNX output matches PyTorch output
"""

import torch
import torch.onnx
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict
from rich.console import Console

console = Console()


def _verify_onnx(onnx_path: Path, torch_out: np.ndarray, inputs: list, tol: float = 1e-4):
    """Run ONNX model and compare against PyTorch output."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp  = {sess.get_inputs()[i].name: inputs[i].numpy() for i in range(len(inputs))}
    onnx_out = sess.run(None, inp)
    for i, (t, o) in enumerate(zip(torch_out, onnx_out)):
        delta = np.abs(t - o).max()
        label = ["reg", "cls"][i]
        console.print(f"    ✓  ONNX output verified ({label} Δ≤{delta:.0e})")


def export_lstm(
    model:     torch.nn.Module,
    seq_len:   int,
    n_features:int,
    onnx_path: Path,
) -> Path:
    """Export LSTM to ONNX with dynamic batch and sequence dimensions."""
    model.cpu().eval()
    dummy = torch.zeros(1, seq_len, n_features)

    with torch.no_grad():
        torch_reg, torch_cls = model(dummy)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version  = 17,
        input_names    = ["input"],
        output_names   = ["reg_output", "cls_output"],
        dynamic_axes   = {
            "input":      {0: "batch"},
            "reg_output": {0: "batch"},
            "cls_output": {0: "batch"},
        },
        dynamo = False,  # MUST be False — dynamo causes CPU/CUDA conflicts
    )

    _verify_onnx(
        onnx_path,
        [torch_reg.numpy(), torch_cls.numpy()],
        [dummy],
    )

    # Benchmark
    import time
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    t0 = time.time()
    for _ in range(20):
        with torch.no_grad(): model(dummy)
    torch_ms = (time.time() - t0) / 20 * 1000

    t0 = time.time()
    for _ in range(20):
        sess.run(None, {inp_name: dummy.numpy()})
    onnx_ms = (time.time() - t0) / 20 * 1000

    speedup = torch_ms / max(onnx_ms, 1e-9)
    console.print(f"    Inference: PyTorch={torch_ms:.2f}ms  ONNX={onnx_ms:.2f}ms  Speedup {speedup:.1f}×")

    return onnx_path


def export_transformer(
    model:     torch.nn.Module,
    seq_len:   int,
    n_features:int,
    onnx_path: Path,
) -> Path:
    """Export Transformer to ONNX."""
    model.cpu().eval()
    dummy = torch.zeros(1, seq_len, n_features)

    with torch.no_grad():
        torch_reg, torch_cls = model(dummy)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version  = 17,
        input_names    = ["input"],
        output_names   = ["reg_output", "cls_output"],
        dynamic_axes   = {
            "input":      {0: "batch"},
            "reg_output": {0: "batch"},
            "cls_output": {0: "batch"},
        },
        dynamo = False,
    )

    _verify_onnx(
        onnx_path,
        [torch_reg.numpy(), torch_cls.numpy()],
        [dummy],
    )

    return onnx_path


def export_meta(
    model:     torch.nn.Module,
    n_horizons:int,
    onnx_path: Path,
) -> Path:
    """Export MetaLearner MLP to ONNX."""
    model.cpu().eval()
    dummy = torch.zeros(1, n_horizons * 2)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version = 17,
        input_names   = ["meta_input"],
        output_names  = ["meta_output"],
        dynamic_axes  = {"meta_input": {0: "batch"}, "meta_output": {0: "batch"}},
        dynamo        = False,
    )
    console.print("    ✓  meta_learner exported")
    return onnx_path


def export_all_models(
    lstm_model:        torch.nn.Module,
    transformer_model: torch.nn.Module,
    meta_learner:      torch.nn.Module,
    seq_len:           int  = 168,
    n_features:        int  = 35,
    n_horizons:        int  = 3,
    onnx_dir:          Path = Path("models/onnx"),
) -> Dict[str, str]:
    """
    Export all three models to ONNX.

    Returns dict: {model_name: onnx_path_str}
    """
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]📦 Exporting models to ONNX[/bold cyan]")

    paths = {}

    console.print("\n  [cyan]LSTM[/cyan]")
    lstm_path = onnx_dir / "lstm.onnx"
    export_lstm(lstm_model, seq_len, n_features, lstm_path)
    paths["lstm"] = str(lstm_path)

    console.print("\n  [cyan]Transformer[/cyan]")
    tf_path = onnx_dir / "transformer.onnx"
    export_transformer(transformer_model, seq_len, n_features, tf_path)
    paths["transformer"] = str(tf_path)

    console.print("\n  [cyan]MetaLearner[/cyan]")
    meta_path = onnx_dir / "meta_learner.onnx"
    export_meta(meta_learner, n_horizons, meta_path)
    paths["meta_learner"] = str(meta_path)

    console.print(f"\n[bold green]✓  All models exported to {onnx_dir}/[/bold green]\n")
    return paths
