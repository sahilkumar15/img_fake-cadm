#!/usr/bin/env python3
"""
eval_all.py
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import model as model_module
from dataset import DeepfakeDataset


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path, net, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network" in ckpt:
        net.load_state_dict(ckpt["network"])
    else:
        net.load_state_dict(ckpt)
    return net


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def get_video_scores(frame_labels, video_names, frame_scores):
    video_dict = {}
    for y, v, s in zip(frame_labels, video_names, frame_scores):
        if v not in video_dict:
            video_dict[v] = {"scores": [], "label": y}
        video_dict[v]["scores"].append(s)

    labels, scores = [], []
    for v in video_dict:
        labels.append(video_dict[v]["label"])
        scores.append(np.mean(video_dict[v]["scores"]))

    return labels, scores


def build_dataset_cfg(master_cfg, ds_entry):
    return {
        "crop_face": master_cfg.get("crop_face", {}),
        "adm_det": master_cfg.get("adm_det", {}),
        "sliding_win": master_cfg.get("sliding_win", {}),
        "dataset": {
            "img_path": ds_entry["img_path"],
            "ld_path": ds_entry["ld_path"],
            "name": ds_entry["name"],
        },
        "model": master_cfg["model"],
        "test": {"batch_size": ds_entry["batch_size"]},
    }


def evaluate_dataset(net, device, ds_cfg, ds_name, out_dir):
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {ds_name}")
    print(f"  img_path  : {ds_cfg['dataset']['img_path']}")
    print(f"{'─'*60}")

    try:
        dataset = DeepfakeDataset("test", ds_cfg)
    except Exception as e:
        print(f"  [SKIP] Failed to load dataset: {e}")
        return None

    if len(dataset) == 0:
        print("  [SKIP] Dataset has 0 samples.")
        return None

    loader = DataLoader(
        dataset,
        batch_size=ds_cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    frame_preds = []
    frame_labels = []
    video_names = []

    net.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc=f"  {ds_name}", leave=False):
            labels, v_names = batch_labels
            labels = labels.long().to(device)
            batch_data = batch_data.to(device)

            outputs = net(batch_data)

            # safe handling: logits shape [B,2] or [B]
            if outputs.ndim == 2 and outputs.shape[1] >= 2:
                scores = outputs[:, 1]
            elif outputs.ndim == 1:
                scores = outputs
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")

            frame_preds.extend(scores.detach().cpu().numpy().tolist())
            frame_labels.extend(labels.detach().cpu().numpy().tolist())
            video_names.extend(list(v_names))

    if len(set(frame_labels)) < 2:
        print("  [SKIP] Only one class present; cannot compute ROC metrics.")
        return None

    f_auc = roc_auc_score(frame_labels, frame_preds)
    f_eer, f_thr = compute_eer(frame_labels, frame_preds)

    v_labels, v_scores = get_video_scores(frame_labels, video_names, frame_preds)
    if len(set(v_labels)) < 2:
        print("  [SKIP] Only one class present at video level; cannot compute ROC metrics.")
        return None

    v_auc = roc_auc_score(v_labels, v_scores)
    v_eer, v_thr = compute_eer(v_labels, v_scores)

    result = {
        "dataset": ds_name,
        "n_frames": len(frame_labels),
        "n_videos": len(v_labels),
        "frame_auc": round(f_auc, 4),
        "frame_eer": round(f_eer, 4),
        "frame_thr": round(f_thr, 6),
        "video_auc": round(v_auc, 4),
        "video_eer": round(v_eer, 4),
        "video_thr": round(v_thr, 6),
    }

    print(f"\n  {'Dataset':<20} {ds_name}")
    print(f"  {'Frames':<20} {len(frame_labels)}  |  Videos: {len(v_labels)}")
    print(f"  {'Frame-AUC':<20} {f_auc:.4f}")
    print(f"  {'Frame-EER':<20} {f_eer:.4f}  (thr={f_thr:.6f})")
    print(f"  {'Video-AUC':<20} {v_auc:.4f}")
    print(f"  {'Video-EER':<20} {v_eer:.4f}  (thr={v_thr:.6f})")

    score_path = os.path.join(out_dir, f"{ds_name.replace('/', '_')}_scores.txt")
    with open(score_path, "w") as f:
        f.write(f"# {ds_name} — frame-level scores\n")
        f.write(f"# frame_auc={f_auc:.4f}  frame_eer={f_eer:.4f}\n")
        f.write(f"# video_auc={v_auc:.4f}  video_eer={v_eer:.4f}\n")
        f.write("label,score,video_name\n")
        for lbl, sc, vn in zip(frame_labels, frame_preds, video_names):
            f.write(f"{lbl},{sc:.6f},{vn}\n")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./configs/eval_config.yaml")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    exp = cfg["experiment"]
    out_dir = os.path.join("experiments", exp["name"], exp["eval_name"])
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Experiment : {exp['name']} / {exp['eval_name']}")
    print(f"  Backbone   : {cfg['model']['backbone']}")
    print(f"  Checkpoint : {cfg['model']['ckpt']}")
    print(f"  Output dir : {out_dir}")
    print(f"  Datasets   : {len(cfg['datasets'])}")
    print(f"{'═'*60}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")

    net = model_module.get(backbone=cfg["model"]["backbone"])
    net = net.to(device)

    if not os.path.exists(cfg["model"]["ckpt"]):
        print(f"\n[ERROR] Checkpoint not found: {cfg['model']['ckpt']}")
        sys.exit(1)

    net = load_checkpoint(cfg["model"]["ckpt"], net, device)
    print(f"  Loaded checkpoint: {cfg['model']['ckpt']}")

    all_results = []
    for ds_entry in cfg["datasets"]:
        ds_cfg = build_dataset_cfg(cfg, ds_entry)
        result = evaluate_dataset(net, device, ds_cfg, ds_entry["name"], out_dir)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n[ERROR] No datasets evaluated successfully.")
        sys.exit(1)

    avg = {
        "dataset": "AVERAGE",
        "n_frames": sum(r["n_frames"] for r in all_results),
        "n_videos": sum(r["n_videos"] for r in all_results),
        "frame_auc": round(np.mean([r["frame_auc"] for r in all_results]), 4),
        "frame_eer": round(np.mean([r["frame_eer"] for r in all_results]), 4),
        "frame_thr": "-",
        "video_auc": round(np.mean([r["video_auc"] for r in all_results]), 4),
        "video_eer": round(np.mean([r["video_eer"] for r in all_results]), 4),
        "video_thr": "-",
    }

    csv_path = os.path.join(out_dir, "results.csv")
    fieldnames = [
        "dataset", "n_frames", "n_videos",
        "frame_auc", "frame_eer", "frame_thr",
        "video_auc", "video_eer", "video_thr"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow(avg)
    print(f"\n  ✓ CSV saved: {csv_path}")

    summary_lines = []
    summary_lines.append(f"\n{'═'*60}")
    summary_lines.append("  EVALUATION SUMMARY")
    summary_lines.append(f"  Experiment : {exp['name']} / {exp['eval_name']}")
    summary_lines.append(f"  Backbone   : {cfg['model']['backbone']}")
    summary_lines.append(f"  Checkpoint : {cfg['model']['ckpt']}")
    summary_lines.append(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"{'─'*60}")
    summary_lines.append(f"  {'Dataset':<20} {'F-AUC':>7} {'F-EER':>7} {'V-AUC':>7} {'V-EER':>7}")
    summary_lines.append(f"  {'─'*56}")

    for r in all_results:
        summary_lines.append(
            f"  {r['dataset']:<20} "
            f"{r['frame_auc']:>7.4f} "
            f"{r['frame_eer']:>7.4f} "
            f"{r['video_auc']:>7.4f} "
            f"{r['video_eer']:>7.4f}"
        )

    summary_lines.append(f"  {'─'*56}")
    summary_lines.append(
        f"  {'AVERAGE':<20} "
        f"{avg['frame_auc']:>7.4f} "
        f"{avg['frame_eer']:>7.4f} "
        f"{avg['video_auc']:>7.4f} "
        f"{avg['video_eer']:>7.4f}"
    )
    summary_lines.append(f"{'═'*60}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"  ✓ Summary saved: {summary_path}")
    print(f"  ✓ All results in: {out_dir}\n")


if __name__ == "__main__":
    main()