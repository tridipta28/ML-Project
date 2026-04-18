"""
================================================================================
run_pipeline.py — Master Orchestration Script
================================================================================
Full end-to-end pipeline:
  1.  Dataset setup & dataloaders
  2.  Model instantiation (one or all architectures)
  3.  Training with mixed precision
  4.  Evaluation: mIoU, Dice, PA, Hausdorff
  5.  Visualization: overlays and comparison plots
  6.  Critical analysis logging
  7.  ONNX export for deployment
  8.  LaTeX report skeleton generation

Usage:
  python run_pipeline.py --model segformer_b2 --epochs 100 --data /data/cityscapes
  python run_pipeline.py --eval_only --ckpt checkpoints/best.pth
  python run_pipeline.py --compare_all          # train+eval all 5 models
================================================================================
"""

import argparse
import json
from pathlib import Path

import torch

# ── Project modules ───────────────────────────────────────────────────────────
from dataset  import get_dataloaders, print_download_instructions, NUM_CLASSES
from models   import build_model, MODEL_REGISTRY
from training import (CombinedLoss, build_optimizer, build_scheduler,
                      Trainer, DEFAULT_CONFIG, compute_class_weights)
from evaluation import (evaluate_model, plot_model_comparison,
                         plot_training_curves, runtime_vs_accuracy_plot)
from extensions import (export_to_onnx, export_to_torchscript,
                         generate_latex_metrics_table,
                         generate_full_report_skeleton,
                         generate_references_bib)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous Driving Segmentation Pipeline"
    )
    p.add_argument("--data",       type=str,  default="/data/cityscapes",
                   help="Path to Cityscapes root directory")
    p.add_argument("--model",      type=str,  default="segformer_b2",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Model architecture to train/evaluate")
    p.add_argument("--epochs",     type=int,  default=100)
    p.add_argument("--batch_size", type=int,  default=4)
    p.add_argument("--img_h",      type=int,  default=512)
    p.add_argument("--img_w",      type=int,  default=1024)
    p.add_argument("--lr",         type=float, default=6e-5)
    p.add_argument("--optimizer",  type=str,  default="adamw",
                   choices=["adam", "adamw", "sgd"])
    p.add_argument("--scheduler",  type=str,  default="warmup_cosine",
                   choices=["cosine", "poly", "step", "warmup_cosine"])
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training, load checkpoint and evaluate")
    p.add_argument("--ckpt",       type=str,  default=None,
                   help="Checkpoint path for eval_only mode")
    p.add_argument("--compare_all", action="store_true",
                   help="Train and evaluate ALL architectures sequentially")
    p.add_argument("--export_onnx", action="store_true",
                   help="Export best model to ONNX after training")
    p.add_argument("--no_amp",     action="store_true",
                   help="Disable mixed precision training")
    p.add_argument("--output_dir", type=str,  default="outputs",
                   help="Directory for all outputs, plots, checkpoints")
    p.add_argument("--workers",    type=int,  default=4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args, device: torch.device) -> dict:
    """Train (optional) and evaluate a single model. Returns metrics dict."""
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Dataloaders ────────────────────────────────────────────────────
    loaders = get_dataloaders(
        root=args.data,
        img_h=args.img_h, img_w=args.img_w,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── 2. Model ──────────────────────────────────────────────────────────
    model = build_model(args.model, device)

    # ── 3. Loss ───────────────────────────────────────────────────────────
    print("[Pipeline] Computing class weights for balanced CE loss...")
    try:
        class_weights = compute_class_weights(
            loaders["train"].dataset, NUM_CLASSES
        ).to(device)
    except Exception:
        class_weights = None
        print("[Pipeline] Skipping class weight computation (dataset unavailable).")

    criterion = CombinedLoss(
        w_ce=1.0, w_dice=0.5, w_focal=0.5,
        class_weights=class_weights,
    )

    if not args.eval_only:
        # ── 4. Optimizer & Scheduler ──────────────────────────────────────
        opt_cfg = {
            "type":            args.optimizer,
            "lr":              args.lr,
            "weight_decay":    1e-2 if args.optimizer == "adamw" else 5e-4,
            "backbone_lr_mult": 0.1,
            "scheduler":       args.scheduler,
            "warmup_epochs":   5,
        }
        optimizer  = build_optimizer(model, opt_cfg)
        scheduler  = build_scheduler(optimizer, opt_cfg, args.epochs)

        # ── 5. Train ──────────────────────────────────────────────────────
        trainer = Trainer(
            model=model, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            device=device,
            log_dir=str(out_dir / "tb_logs"),
            ckpt_dir=str(out_dir / "checkpoints"),
            use_amp=not args.no_amp,
        )

        if args.ckpt:
            trainer.load_checkpoint(args.ckpt)

        history = trainer.train(loaders["train"], loaders["val"], args.epochs)

        # Plot training curves
        plot_training_curves(history, save_path=str(out_dir / "training_curves.png"))

        # Load best weights for evaluation
        best_ckpt = out_dir / "checkpoints" / "best.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[Pipeline] Loaded best checkpoint (mIoU={ckpt['miou']:.4f})")

    else:
        # ── Eval-only: load checkpoint ────────────────────────────────────
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[Pipeline] Loaded {args.ckpt}")
        else:
            print("[Pipeline] WARNING: eval_only with no checkpoint — using random weights.")

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    results = evaluate_model(
        model=model,
        loader=loaders["val"],
        device=device,
        model_name=args.model,
        save_dir=str(out_dir),
        compute_hd=True,
        n_vis=8,
    )

    # ── 7. ONNX Export ────────────────────────────────────────────────────
    if args.export_onnx:
        export_to_onnx(model,
                       save_path=str(out_dir / f"{args.model}.onnx"),
                       img_h=args.img_h, img_w=args.img_w)
        export_to_torchscript(model,
                              save_path=str(out_dir / f"{args.model}_scripted.pt"),
                              img_h=args.img_h, img_w=args.img_w)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Compare all models
# ─────────────────────────────────────────────────────────────────────────────

def run_compare_all(args, device: torch.device):
    """Train and evaluate all architectures; generate comparison report."""
    all_results = {}
    for model_name in MODEL_REGISTRY:
        print(f"\n{'═'*60}")
        print(f"  PIPELINE: {model_name.upper()}")
        print(f"{'═'*60}")
        args.model = model_name
        try:
            results = run_single(args, device)
            all_results[model_name] = results
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            continue

    # ── Comparison plots ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    plot_model_comparison(all_results,
                          save_path=str(out_dir / "model_comparison.png"))

    # Runtime vs accuracy scatter
    names  = list(all_results.keys())
    mious  = [all_results[m].get("mIoU",           0) * 100 for m in names]
    fps    = [all_results[m].get("fps",             1)       for m in names]
    # Approximate parameter counts (millions)
    params = {"unetplusplus": 31, "deeplabv3plus": 59, "pspnet": 46,
              "segformer_b0": 3.7, "segformer_b2": 27, "mask2former": 44}
    param_vals = [params.get(m, 30) for m in names]

    runtime_vs_accuracy_plot(names, mious, fps, param_vals,
                              save_path=str(out_dir / "efficiency_plot.png"))

    # ── LaTeX report ──────────────────────────────────────────────────────
    generate_latex_metrics_table(all_results,
                                  save_path=str(out_dir / "metrics_table.tex"))
    generate_full_report_skeleton(save_path=str(out_dir / "report_skeleton.tex"))
    generate_references_bib(save_path=str(out_dir / "references.bib"))

    # ── Save results JSON ─────────────────────────────────────────────────
    with open(str(out_dir / "all_results.json"), "w") as f:
        # iou_per_class is a list of floats — serialisable
        json.dump(all_results, f, indent=2)

    print(f"\n{'═'*60}")
    print("  ALL RESULTS SUMMARY")
    print(f"{'═'*60}")
    for name, res in all_results.items():
        print(f"  {name:<20} mIoU={res.get('mIoU',0):.4f}  "
              f"FPS={res.get('fps',0):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'═'*60}\n")

    # Check dataset exists
    if not Path(args.data).exists():
        print_download_instructions()
        print(f"\n[ERROR] Dataset not found at: {args.data}")
        print("Set --data to your Cityscapes root directory and re-run.")
        exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        run_compare_all(args, device)
    else:
        results = run_single(args, device)
        print("\n[Final Results]")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
