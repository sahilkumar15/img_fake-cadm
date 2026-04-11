#!/usr/bin/env python3
# dataset.py  — generalization-safe version
#
# FIX:
#   - In TRAIN mode: still require source image.
#   - In TEST mode: do NOT drop sample if source image is missing.
#     Instead, fall back to target image path.
#   - This allows source-free generalization evaluation.
#
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input


class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    Args:
        mode: 'train' or 'test'
        config: config dict
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        self.root = self.config['dataset']['img_path']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random

        assert mode in ['train', 'test']
        self.do_train = (mode == 'train')

        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

        print(f"  [{mode}] {len(self.samples)} valid samples loaded.")

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def collect_samples(self) -> List:
        samples = []

        skipped_no_key = 0
        skipped_no_img = 0
        skipped_no_src = 0

        directory = os.path.expanduser(self.root)

        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue

            for r, _, filenames in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filenames):

                    # only image files
                    if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    path = os.path.join(r, name)

                    # strip extension to match ldm.json key
                    info_key = os.path.splitext(path)[0]

                    # Guard 1: sample must exist in ldm.json
                    if info_key not in self.info_meta_dict:
                        skipped_no_key += 1
                        continue

                    info_meta = self.info_meta_dict[info_key]

                    # Guard 2: target image must exist
                    if not os.path.isfile(path):
                        skipped_no_img += 1
                        continue

                    # metadata
                    landmark = info_meta['landmark']
                    class_label = int(info_meta['label'])

                    # safer video name
                    video_name = os.path.dirname(path)

                    # source path from metadata
                    source_base = info_meta.get('source_path', None)

                    if source_base is None:
                        # TRAIN: skip
                        # TEST : fallback to target image
                        if self.mode == 'train':
                            skipped_no_src += 1
                            continue
                        else:
                            source_path = path
                            skipped_no_src += 1
                    else:
                        source_path = source_base + os.path.splitext(path)[1]

                        # if relative path is stored, resolve it relative to root
                        if not os.path.isabs(source_path):
                            source_path = os.path.normpath(source_path)

                        if not os.path.isfile(source_path):
                            if self.mode == 'train':
                                skipped_no_src += 1
                                continue
                            else:
                                # source-free fallback at test time
                                source_path = path
                                skipped_no_src += 1

                    samples.append(
                        (
                            path,
                            {
                                'labels': class_label,
                                'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name
                            }
                        )
                    )

        print(
            f"  [dataset] skipped: {skipped_no_key} not-in-ldm, "
            f"{skipped_no_img} missing-img, "
            f"{skipped_no_src} missing-source"
        )

        return samples

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]

        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)

        # Final safety net
        if img is None:
            return None, "target imread returned None"

        if source_img is None:
            # only fallback in test mode
            if self.mode == 'test':
                source_img = img.copy()
            else:
                return None, "source imread returned None"

        if self.mode == "train":
            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )

            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = torch.Tensor(img.transpose(2, 0, 1))

            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']

            return img, (label, video_name)

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from lib.util import load_config
    config = load_config('./configs/train_ffpp.cfg')
    d = DeepfakeDataset(mode="train", config=config)
    print(f"Total samples: {len(d)}")