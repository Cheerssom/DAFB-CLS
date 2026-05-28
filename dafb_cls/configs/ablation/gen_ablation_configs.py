import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))

def load_base(name):
    with open(os.path.join(BASE_DIR, name), "r") as f:
        return yaml.safe_load(f)

def save_cfg(name, cfg):
    path = os.path.join(ABLATION_DIR, name)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  -> {path}")

def make_variant(base_name, variant_name, overrides):
    base = load_base(base_name)
    base["save_dir"] = f"./checkpoints/ablation/{base_name.replace('.yaml','')}/{variant_name}"
    for k, v in overrides.items():
        if k == "layer_indices":
            base["layer_indices"] = v
        elif k.startswith("ablation."):
            key = k.split(".", 1)[1]
            if "ablation" not in base:
                base["ablation"] = {}
            base["ablation"][key] = v
        else:
            base[k] = v
    filename = f"{base_name.replace('.yaml','')}_{variant_name}.yaml"
    save_cfg(filename, base)

def main():
    os.makedirs(ABLATION_DIR, exist_ok=True)

    # ============================================================
    # OpenCLIP incremental ablation
    # ============================================================
    print("\n=== OpenCLIP Incremental Ablation ===")

    make_variant("openclip_vitb16_voc.yaml", "baseline_clip", {
        "ablation.disable_dafb": True,
        "ablation.lastvit_mode": "dual",
    })

    make_variant("openclip_vitb16_voc.yaml", "no_budget", {
        "ablation.disable_adaptive_budget": True,
        "ablation.budget_fixed_ratio": 0.3,
    })

    make_variant("openclip_vitb16_voc.yaml", "no_dual_cls", {
        "ablation.disable_dual_cls": True,
    })

    make_variant("openclip_vitb16_voc.yaml", "shared_depth", {
        "ablation.share_depth_attn": True,
    })

    make_variant("openclip_vitb16_voc.yaml", "hard_topk", {
        "ablation.mask_type": "hard_topk",
        "ablation.topk_ratio": 0.3,
    })

    make_variant("openclip_vitb16_voc.yaml", "full", {})

    # ============================================================
    # OpenCLIP component ablation (w/o each cue)
    # ============================================================
    print("\n=== OpenCLIP Component Ablation ===")

    make_variant("openclip_vitb16_voc.yaml", "no_cues", {
        "ablation.disable_cues": True,
    })

    make_variant("openclip_vitb16_voc.yaml", "no_adaptive_budget", {
        "ablation.disable_adaptive_budget": True,
        "ablation.budget_fixed_ratio": 0.3,
    })

    # ============================================================
    # DINO incremental ablation
    # ============================================================
    print("\n=== DINO Incremental Ablation ===")

    make_variant("dino_vits16_voc.yaml", "baseline_dino", {
        "ablation.disable_dafb": True,
        "ablation.lastvit_mode": "dual",
    })

    make_variant("dino_vits16_voc.yaml", "no_budget", {
        "ablation.disable_adaptive_budget": True,
        "ablation.budget_fixed_ratio": 0.3,
    })

    make_variant("dino_vits16_voc.yaml", "no_dual_cls", {
        "ablation.disable_dual_cls": True,
    })

    make_variant("dino_vits16_voc.yaml", "shared_depth", {
        "ablation.share_depth_attn": True,
    })

    make_variant("dino_vits16_voc.yaml", "hard_topk", {
        "ablation.mask_type": "hard_topk",
        "ablation.topk_ratio": 0.3,
    })

    make_variant("dino_vits16_voc.yaml", "full", {})

    print("\nDone! All ablation configs generated.")

if __name__ == "__main__":
    main()
