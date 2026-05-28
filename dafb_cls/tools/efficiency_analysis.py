"""
DAFB-CLS Efficiency Analysis
Reports: parameter counts, FLOPs, GPU memory, inference latency
Usage:
    python -m dafb_cls.tools.efficiency_analysis
    python -m dafb_cls.tools.efficiency_analysis --configs dino_vits16_voc.yaml openclip_vitb16_voc.yaml
"""
import os, sys, time, argparse, json
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn

# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_backbone_params(model):
    if hasattr(model, "extractor"):
        return sum(p.numel() for p in model.extractor.parameters())
    return 0

def count_flops_fvcore(model, input_tensor, description="model"):
    """Try fvcore FlopCountAnalysis; returns (flops_str, flops_raw) or (error_msg, None)."""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, input_tensor)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            raw = flops.total()
            return _fmt_flops(raw), raw
    except Exception as e:
        return f"[fvcore failed: {e}]", None

def count_flops_thop(model, input_tensor):
    """Fallback via thop; returns (flops_str, raw) or (error_msg, None)."""
    try:
        from thop import profile as thop_profile
        model.eval()
        with torch.no_grad():
            macs, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
            flops = macs * 2  # MACs -> FLOPs (multiply-accumulate ≈ 2 ops)
            return _fmt_flops(flops), flops
    except Exception as e:
        return f"[thop failed: {e}]", None

def _fmt_flops(f):
    if f is None:
        return "N/A"
    if f >= 1e12:
        return f"{f/1e12:.2f} TFLOPs"
    if f >= 1e9:
        return f"{f/1e9:.2f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.2f} MFLOPs"
    return f"{f:.0f} FLOPs"

def _fmt_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)

def measure_latency(model, input_tensor, device, warmup=10, repeats=50, label=""):
    """Return (avg_ms, std_ms). Uses CUDA events when available."""
    model.eval()
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

    if is_cuda:
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            model(input_tensor)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
    else:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(input_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    import statistics
    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return avg, std

def measure_peak_memory(model, input_tensor, device):
    """Return peak GPU memory (MB) during a forward pass. Returns 0 on CPU."""
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    with torch.no_grad():
        model(input_tensor)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)

# ── model builders ───────────────────────────────────────────────────────────

def build_full_model(cfg):
    from dafb_cls.models.dafb_cls_model import DAFBCLS
    model = DAFBCLS(cfg)
    return model

def build_lastvit_model(cfg):
    """LaSt-ViT baseline: DAFB-CLS with disable_dafb=True, lastvit_mode='dual'."""
    cfg = dict(cfg)
    cfg["ablation"] = {"disable_dafb": True, "lastvit_mode": "dual"}
    return build_full_model(cfg)

def build_baseline_model(cfg):
    """Raw ViT backbone with a linear head — simplest baseline."""
    from dafb_cls.models.feature_extractor import MultiLayerFeatureExtractor
    backbone_type = cfg.get("backbone_type", "dino_vits16")
    ext = MultiLayerFeatureExtractor(
        backbone_type=backbone_type,
        layer_indices=cfg.get("layer_indices", [3, 6, 9, 12]),
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 16),
    )
    ext.build_backbone()
    dim = ext.dim
    num_classes = cfg.get("num_classes", 20)
    task = cfg.get("task", "object_discovery")

    class BackboneBaseline(nn.Module):
        def __init__(self, ext, dim, num_classes, task):
            super().__init__()
            self.extractor = ext
            if task == "object_discovery":
                self.head = nn.Linear(dim, 1)
            else:
                self.head = nn.Linear(dim, num_classes)
            self.task = task

        def forward(self, x):
            patch_features, cls_feat = self.extractor(x, return_cls=True)
            if cls_feat is not None:
                feat = cls_feat
            else:
                feat = patch_features[:, -1, :, :].mean(dim=1)
            if self.task == "object_discovery":
                return self.head(patch_features[:, -1, :, :]).squeeze(-1)
            return self.head(feat)

    return BackboneBaseline(ext, dim, num_classes, task)

# ── report ───────────────────────────────────────────────────────────────────

def analyze_config(cfg_path, device, repeats=50):
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    backbone  = cfg.get("backbone_type", "unknown")
    image_size = cfg.get("image_size", 224)
    task       = cfg.get("task", "classification")
    bs         = 1  # single-sample latency

    print(f"\n{'='*70}")
    print(f"  Config: {os.path.basename(cfg_path)}")
    print(f"  Backbone: {backbone}  |  Task: {task}  |  Input: {image_size}x{image_size}")
    print(f"{'='*70}")

    dummy = torch.randn(bs, 3, image_size, image_size, device=device)

    rows = []  # for final summary table

    for label, builder in [
        ("Backbone Baseline", build_baseline_model),
        ("LaSt-ViT",          build_lastvit_model),
        ("DAFB-CLS (full)",   build_full_model),
    ]:
        try:
            model = builder(cfg).to(device).eval()
        except Exception as e:
            print(f"\n  [{label}] build failed: {e}")
            continue

        total_params, trainable_params = count_params(model)
        backbone_params = count_backbone_params(model)
        aggregator_params = total_params - backbone_params

        flops_str, flops_raw = count_flops_fvcore(model, dummy, label)
        if flops_raw is None:
            flops_str, flops_raw = count_flops_thop(model, dummy)

        avg_ms, std_ms = measure_latency(model, dummy, device, warmup=10, repeats=repeats, label=label)
        peak_mem = measure_peak_memory(model, dummy, device)

        print(f"\n  ┌─ {label}")
        print(f"  │  Parameters:   total {_fmt_params(total_params)}"
              f"  |  backbone {_fmt_params(backbone_params)}"
              f"  |  trainable {_fmt_params(trainable_params)}")
        print(f"  │  FLOPs/sample: {flops_str}")
        print(f"  │  Latency:      {avg_ms:.2f} ± {std_ms:.2f} ms  (batch=1, {repeats} runs)")
        if peak_mem > 0:
            print(f"  │  Peak GPU mem: {peak_mem:.1f} MB  (forward pass, batch=1)")
        print(f"  └{'─'*60}")

        rows.append({
            "model": label,
            "backbone": backbone,
            "total_params": total_params,
            "backbone_params": backbone_params,
            "trainable_params": trainable_params,
            "aggregator_params": aggregator_params,
            "flops_raw": flops_raw,
            "flops_str": flops_str,
            "latency_ms": avg_ms,
            "latency_std_ms": std_ms,
            "peak_gpu_mem_mb": peak_mem,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows

def print_summary_table(all_rows):
    if not all_rows:
        return
    print(f"\n{'='*90}")
    print("  SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"  {'Config':<30} {'Model':<20} {'Params':>10} {'Agg Params':>12} {'FLOPs':>14} {'Latency':>12} {'GPU Mem':>10}"
    print(header)
    print("  " + "-" * 88)
    for r in all_rows:
        mem_str = f"{r['peak_gpu_mem_mb']:.0f} MB" if r['peak_gpu_mem_mb'] > 0 else "N/A"
        cfg_name = r.get("backbone", "?")
        print(f"  {cfg_name:<30} {r['model']:<20} {_fmt_params(r['total_params']):>10}"
              f"  {_fmt_params(r['aggregator_params']):>10}  {r['flops_str']:>14}"
              f"  {r['latency_ms']:>8.2f} ms  {mem_str:>10}")

    print()

def print_deltas(all_rows):
    """Print speedup and overhead ratios for each config group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_rows:
        groups[r["backbone"]].append(r)

    print(f"{'='*90}")
    print("  OVERHEAD ANALYSIS")
    print(f"{'='*90}")
    header = f"  {'Backbone':<25} {'Baseline':>12} {'LaSt-ViT':>12} {'DAFB-CLS':>12} {'DAFB overhead':>16}"
    print(header)
    print("  " + "-" * 80)

    for backbone, rows in groups.items():
        by_name = {r["model"]: r for r in rows}
        base = by_name.get("Backbone Baseline")
        lastvit = by_name.get("LaSt-ViT")
        dafb = by_name.get("DAFB-CLS (full)")
        if not dafb:
            continue

        def _lat(r):
            return f"{r['latency_ms']:.2f} ms" if r else "N/A"

        overhead = ""
        if base and dafb:
            ratio = dafb["latency_ms"] / base["latency_ms"] if base["latency_ms"] > 0 else float("inf")
            overhead = f"{ratio:.2f}x"

        print(f"  {backbone:<25} {_lat(base):>12} {_lat(lastvit):>12} {_lat(dafb):>12} {overhead:>16}")

    print()

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DAFB-CLS Efficiency Analysis")
    parser.add_argument("--configs", nargs="+", default=[
        "dafb_cls/configs/dino_vits16_voc.yaml",
        "dafb_cls/configs/openclip_vitb16_voc.yaml",
    ], help="Config YAML files to analyze")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=50, help="Latency measurement repeats")
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda or 'N/A'}")
    print(f"Latency: {args.repeats} runs, batch_size=1")

    all_rows = []
    for cfg_path in args.configs:
        if not os.path.exists(cfg_path):
            print(f"WARNING: config not found: {cfg_path}, skipping")
            continue
        rows = analyze_config(cfg_path, device, repeats=args.repeats)
        all_rows.extend(rows)

    print_summary_table(all_rows)
    print_deltas(all_rows)

    if args.json:
        # convert raw values to serializable form
        out = []
        for r in all_rows:
            row = dict(r)
            for k in ("total_params", "backbone_params", "trainable_params", "aggregator_params"):
                row[k] = int(row[k])
            row["flops_raw"] = int(row["flops_raw"]) if row["flops_raw"] else None
            row["latency_ms"] = round(row["latency_ms"], 3)
            row["latency_std_ms"] = round(row["latency_std_ms"], 3)
            row["peak_gpu_mem_mb"] = round(row["peak_gpu_mem_mb"], 1)
            out.append(row)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.json}")

if __name__ == "__main__":
    main()
