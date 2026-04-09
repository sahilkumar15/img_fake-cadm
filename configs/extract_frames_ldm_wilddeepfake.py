

#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
from imutils import face_utils


DATA_PATH = "/scratch/sahil/projects/img_deepfake/datasets/wild_deepfake/test"
SAVE_IMGS_PATH = "./test_images_wilddeepfake"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
IMG_META_DICT = {}

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(files)


def parse_label(img_path):
    path = img_path.lower()
    if "fake" in path or "forged" in path:
        return 1
    if "real" in path or "original" in path:
        return 0
    raise ValueError(f"Cannot infer label from path: {img_path}")


def get_save_path(img_path):
    rel = os.path.relpath(img_path, DATA_PATH)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(SAVE_IMGS_PATH, rel_no_ext)


def get_source_path(save_path, label):
    # WildDeepfake usually does not provide FF++-style source pairing.
    # Use self-path fallback for test-time compatibility.
    return save_path


def make_dlib_ready(frame):
    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None

    return np.ascontiguousarray(frame, dtype=np.uint8)


def preprocess_image(img_path, save_path, face_detector, face_predictor):
    label = parse_label(img_path)
    source_save_path = get_source_path(save_path, label)
    os.makedirs(save_path, exist_ok=True)

    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        tqdm.write(f"Failed to read image: {img_path}")
        return

    frame_rgb = make_dlib_ready(frame)
    if frame_rgb is None:
        tqdm.write(f"Unsupported image type: {img_path}")
        return

    try:
        faces = face_detector(frame_rgb, 1)
    except RuntimeError as e:
        tqdm.write(f"dlib failed on {os.path.basename(img_path)}: {e}")
        return

    if len(faces) == 0:
        tqdm.write(f"No faces in {os.path.basename(img_path)}")
        return

    landmarks = []
    size_list = []

    for face_idx in range(len(faces)):
        landmark = face_predictor(frame_rgb, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks.append(landmark)

    landmarks = np.stack(landmarks, axis=0)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

    key = f"{save_path}/frame_0"
    IMG_META_DICT[key] = {
        "landmark": landmarks.tolist(),
        "source_path": f"{source_save_path}/frame_0",
        "label": label,
    }

    image_path = f"{save_path}/frame_0.png"
    cv2.imwrite(image_path, frame)


def main():
    os.makedirs(SAVE_IMGS_PATH, exist_ok=True)

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Landmark predictor not found: {PREDICTOR_PATH}")

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    all_imgs = list_images(DATA_PATH)
    print(f"Total test images found: {len(all_imgs)}")

    for img_path in tqdm(all_imgs, desc="Processing WildDeepfake"):
        save_path = get_save_path(img_path)
        preprocess_image(img_path, save_path, face_detector, face_predictor)

    with open(f"{SAVE_IMGS_PATH}/ldm.json", "w") as f:
        json.dump(IMG_META_DICT, f)

    print(f"Saved landmark metadata to: {SAVE_IMGS_PATH}/ldm.json")
    print(f"Total processed images with landmarks: {len(IMG_META_DICT)}")


if __name__ == "__main__":
    main()
    
# cd /scratch/sahil/projects/img_deepfake/code_2/CADDM
# rm -rf test_images_wilddeepfake
# mkdir -p test_images_wilddeepfake
# python lib/extract_frames_ldm_wilddeepfake.py
# ls test_images_wilddeepfake/ldm.json