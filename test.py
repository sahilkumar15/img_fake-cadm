#!/usr/bin/env python3
# test.py
import argparse
from sklearn.metrics import roc_auc_score, roc_curve

import torch
from torch.utils.data import DataLoader

import model
from dataset import DeepfakeDataset
from lib.util import load_config


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    return parser.parse_args()


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt, map_location=device)
    net.load_state_dict(checkpoint['network'])
    return net


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = (abs(fnr - fpr)).argmin()
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return eer, threshold


def get_video_scores(frame_labels, video_names, frame_scores):
    video_dict = {}

    for y, v, s in zip(frame_labels, video_names, frame_scores):
        if v not in video_dict:
            video_dict[v] = {"scores": [], "label": y}
        video_dict[v]["scores"].append(s)

    video_labels = []
    video_scores = []

    for v in video_dict:
        video_labels.append(video_dict[v]["label"])
        video_scores.append(sum(video_dict[v]["scores"]) / len(video_dict[v]["scores"]))

    return video_labels, video_scores


def test():
    args = args_func()
    cfg = load_config(args.cfg)

    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()

    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=4,
    )

    frame_pred_list = []
    frame_label_list = []
    video_name_list = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            labels, video_name = batch_labels
            labels = labels.long().to(device)
            batch_data = batch_data.to(device)

            outputs = net(batch_data)
            outputs = outputs[:, 1]

            frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
            frame_label_list.extend(labels.detach().cpu().numpy().tolist())
            video_name_list.extend(list(video_name))

    # Frame-level metrics
    f_auc = roc_auc_score(frame_label_list, frame_pred_list)
    f_eer, f_thr = compute_eer(frame_label_list, frame_pred_list)

    # Video-level metrics
    video_label_list, video_pred_list = get_video_scores(
        frame_label_list, video_name_list, frame_pred_list
    )
    v_auc = roc_auc_score(video_label_list, video_pred_list)
    v_eer, v_thr = compute_eer(video_label_list, video_pred_list)

    print(f"Frame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"Frame-EER of {cfg['dataset']['name']} is {f_eer:.4f} at threshold {f_thr:.6f}")
    print(f"Video-AUC of {cfg['dataset']['name']} is {v_auc:.4f}")
    print(f"Video-EER of {cfg['dataset']['name']} is {v_eer:.4f} at threshold {v_thr:.6f}")


if __name__ == "__main__":
    test()
    
    
# Fix all at once
# python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images/ldm.json          --img_path ./test_images-2/test_images
# python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_celebdf/ldm.json   --img_path ./test_images-2/test_images_celebdf
# python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_dfd/ldm.json       --img_path ./test_images-2/test_images_dfd
# python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_diffswap/ldm.json  --img_path ./test_images-2/test_images_diffswap
# python lib/fix_ldm_paths.py --ldm ./test_images-2/test_images_wilddeepfake/ldm.json --img_path ./test_images-2/test_images_wilddeepfake
    
# CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test.cfg
# CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test_celebdf.cfg
# CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test_dfd.cfg
# CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test_diffswap.cfg
# CUDA_VISIBLE_DEVICES=1 python test.py --cfg ./configs/caddm_test_wilddeepfake.cfg

