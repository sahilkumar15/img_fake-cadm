#!/usr/bin/env python3
# lib/ff++_train.py
#
# THE ACTUAL ROOT CAUSE (confirmed by diagnose.py output):
#   pix_fmt="gbrp" — ALL raw FF++ videos use GBR planar pixel format.
#   OpenCV's VideoCapture reads gbrp frames as uint8 (H,W,3) arrays but the
#   channel data is in wrong order / wrong memory layout. dlib's numpy_image.h
#   does a strict internal check and rejects 100% of these frames.
#   This affects every single video — it is not a "some videos fail" issue.
#
# THE FIX:
#   Use ffmpeg as a pipe to decode frames to raw rgb24 bytes, then read those
#   bytes directly into numpy. This completely bypasses OpenCV's VideoCapture
#   for reading — ffmpeg handles gbrp→rgb24 conversion correctly internally.
#   Everything else (dlib, landmark detection, saving PNGs, ldm.json) is
#   identical to the original working script.
#
# USAGE:
#   cd /scratch/sahil/projects/img_deepfake/code_2/img_fake-cadm
#   python lib/lib/ff++_train.py
#
# REQUIRES: ffmpeg on PATH
#   Check:  ffmpeg -version
#   Load:   module load ffmpeg        (on Katz HPC)
#    or:    conda install -c conda-forge ffmpeg
# ─────────────────────────────────────────────────────────────────────────────

from glob import glob
import os
import cv2
import subprocess
import shutil
from tqdm import tqdm
import numpy as np
import dlib
import json
from imutils import face_utils


# ── Config ────────────────────────────────────────────────────────────────────

VIDEO_PATH      = "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
TRAIN_SAVE_PATH = "./dataset_images-3/ffpp/train_images"
TEST_SAVE_PATH  = "./dataset_images-3/ffpp/test_images"
PREDICTOR_PATH  = "./lib/shape_predictor_81_face_landmarks.dat"
SPLIT_CSV_PATH  = f"{VIDEO_PATH}/split_csv_4k"

DATASETS    = ["Original", "Deepfakes", "Face2Face",
               "FaceSwap", "NeuralTextures", "FaceShifter"]
COMPRESSION = ["raw"]
NUM_FRAMES  = 10   # frames per video (use 1 to match original test script speed)

# ─────────────────────────────────────────────────────────────────────────────


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "\nffmpeg not found. Please run ONE of:\n"
            "  module load ffmpeg              (Katz HPC)\n"
            "  conda install -c conda-forge ffmpeg\n"
            "  sudo apt-get install ffmpeg\n"
        )
    print(f"[ok] ffmpeg found: {shutil.which('ffmpeg')}")


def get_video_info(video_path):
    """Get (n_frames, width, height) via ffprobe. Falls back to OpenCV."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,width,height",
            "-of", "csv=p=0",
            video_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        # output: "nb_frames,width,height"  or  "width,height,nb_frames" depending on ffprobe version
        # Use safer approach: parse key=value
        cmd2 = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,width,height",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        out2 = subprocess.check_output(cmd2, stderr=subprocess.DEVNULL).decode()
        info = {}
        for line in out2.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        n = int(info.get("nb_frames", 0))
        w = int(info.get("width",  640))
        h = int(info.get("height", 480))
        if n > 0:
            return n, w, h
    except Exception:
        pass

    # OpenCV fallback (may under-count for gbrp)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return max(n, 1), w, h


def read_frames_ffmpeg(video_path, frame_idxs, width, height):
    """
    Use ffmpeg to decode only the needed frames from video_path.
    Yields (frame_idx, rgb_uint8_ndarray) for each successfully decoded frame.

    Strategy: pipe the full video as raw rgb24 frames, extract only the
    frames at frame_idxs. This avoids OpenCV VideoCapture entirely.
    """
    frame_idxs_set = set(int(i) for i in frame_idxs)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",   # KEY: convert gbrp → packed rgb24
        "-",                   # output to stdout pipe
    ]

    frame_size = width * height * 3
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_size * 4,
        )

        cnt = 0
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break   # end of video

            if cnt in frame_idxs_set:
                # Parse raw rgb24 bytes → numpy RGB array
                frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                frame_rgb = np.ascontiguousarray(frame_rgb)
                yield cnt, frame_rgb

                # Early exit once we have all needed frames
                frame_idxs_set.discard(cnt)
                if not frame_idxs_set:
                    break

            cnt += 1

    except Exception as e:
        tqdm.write(f"[ffmpeg pipe error] {os.path.basename(video_path)}: {e}")
    finally:
        if proc:
            proc.stdout.close()
            proc.wait()


def save_frame_as_png(frame_rgb, out_path):
    """Save an RGB numpy array as PNG (convert to BGR for OpenCV imwrite)."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, frame_bgr)


# ── Split loading ─────────────────────────────────────────────────────────────

def load_split_ids():
    train_json = os.path.join(SPLIT_CSV_PATH, "train.json")
    val_json   = os.path.join(SPLIT_CSV_PATH, "val.json")
    test_json  = os.path.join(SPLIT_CSV_PATH, "test.json")

    if os.path.exists(train_json) and os.path.exists(test_json):
        with open(train_json) as f:
            train_ids = {v for pair in json.load(f) for v in pair}
        with open(test_json) as f:
            test_ids = {v for pair in json.load(f) for v in pair}
        if os.path.exists(val_json):
            with open(val_json) as f:
                test_ids |= {v for pair in json.load(f) for v in pair}
        print(f"[split] CSVs: {len(train_ids)} train / {len(test_ids)} test IDs")
        return train_ids, test_ids
    else:
        train_ids = {f"{i:03d}" for i in range(720)}
        test_ids  = {f"{i:03d}" for i in range(720, 1000)}
        print(f"[split] index split: {len(train_ids)} train / {len(test_ids)} test")
        return train_ids, test_ids


# ── Path helpers ──────────────────────────────────────────────────────────────

def parse_video_path(dataset, compression):
    if dataset == "Original":
        path = f"{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/"
    elif dataset in ["FaceShifter", "Face2Face", "Deepfakes", "FaceSwap", "NeuralTextures"]:
        path = f"{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/"
    else:
        raise NotImplementedError
    movies = sorted(glob(path + "*.mp4"))
    print(f"  {len(movies)} videos in {dataset}/{compression}")
    return movies


def get_video_base_id(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    return name.split("_")[0]


def parse_labels(video_path):
    return 0 if "original" in video_path else 1


def parse_source_save_path(save_path, save_root):
    """Identical logic to original working script."""
    if "original" in save_path:
        return save_path
    img_meta            = save_path.split("/")
    source_target_index = img_meta[-1]
    source_index        = source_target_index.split("_")[0]
    manipulation_name   = img_meta[-4]
    return (
        save_path
        .replace("manipulated_sequences", "original_sequences")
        .replace(manipulation_name, "youtube")
        .replace(source_target_index, source_index)
    )


# ── Main processing ───────────────────────────────────────────────────────────

def preprocess_video(video_path, save_path, source_save_path,
                     face_detector, face_predictor, img_meta_dict):
    label = parse_labels(video_path)
    os.makedirs(save_path, exist_ok=True)

    n_frames, width, height = get_video_info(video_path)
    if n_frames <= 0:
        tqdm.write(f"[warn] 0 frames: {os.path.basename(video_path)}")
        return

    frame_idxs = np.linspace(0, n_frames - 1, min(NUM_FRAMES, n_frames),
                              endpoint=True, dtype=int)

    for cnt_frame, frame_rgb in read_frames_ffmpeg(video_path, frame_idxs, width, height):
        # frame_rgb is already proper uint8 RGB from ffmpeg pipe
        # dlib requires contiguous uint8 RGB — already satisfied
        try:
            faces = face_detector(frame_rgb, 1)
        except RuntimeError as e:
            tqdm.write(f"dlib error frame {cnt_frame} {os.path.basename(video_path)}: {e}")
            continue

        if len(faces) == 0:
            continue

        # Pick largest face
        landmarks = []
        size_list = []
        for face in faces:
            lm = face_predictor(frame_rgb, face)
            lm = face_utils.shape_to_np(lm)
            x0, y0 = lm[:, 0].min(), lm[:, 1].min()
            x1, y1 = lm[:, 0].max(), lm[:, 1].max()
            size_list.append((x1 - x0) * (y1 - y0))
            landmarks.append(lm)

        landmarks = np.stack(landmarks, axis=0)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

        key = f"{save_path}/frame_{cnt_frame}"
        img_meta_dict[key] = {
            "landmark":    landmarks.tolist(),
            "source_path": f"{source_save_path}/frame_{cnt_frame}",
            "label":       label,
        }

        save_frame_as_png(frame_rgb, f"{key}.png")


def main():
    check_ffmpeg()

    os.makedirs(TRAIN_SAVE_PATH, exist_ok=True)
    os.makedirs(TEST_SAVE_PATH,  exist_ok=True)

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Predictor not found: {PREDICTOR_PATH}")

    face_detector  = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    train_ids, test_ids = load_split_ids()
    train_meta = {}
    test_meta  = {}

    for dataset in DATASETS:
        for comp in COMPRESSION:
            try:
                movies = parse_video_path(dataset, comp)
            except NotImplementedError:
                continue
            if not movies:
                continue

            for i in tqdm(range(len(movies)), desc=f"{dataset}-{comp}"):
                video_path = movies[i]
                base_id    = get_video_base_id(video_path)

                if base_id in test_ids:
                    save_root = TEST_SAVE_PATH
                    meta_dict = test_meta
                else:
                    save_root = TRAIN_SAVE_PATH
                    meta_dict = train_meta

                save_path = (
                    video_path
                    .replace(VIDEO_PATH, save_root)
                    .replace(".mp4", "")
                    .replace("/videos", "/frames")
                )
                source_save_path = parse_source_save_path(save_path, save_root)

                preprocess_video(
                    video_path, save_path, source_save_path,
                    face_detector, face_predictor, meta_dict
                )

    with open(f"{TRAIN_SAVE_PATH}/ldm.json", "w") as f:
        json.dump(train_meta, f)
    with open(f"{TEST_SAVE_PATH}/ldm.json", "w") as f:
        json.dump(test_meta, f)

    print(f"\n✓ Done!")
    print(f"  Train: {len(train_meta)} frames → {TRAIN_SAVE_PATH}/ldm.json")
    print(f"  Test:  {len(test_meta)} frames  → {TEST_SAVE_PATH}/ldm.json")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# RUN:
#   # Check ffmpeg first:
#   ffmpeg -version   (if missing: module load ffmpeg)
#
#   # Extract frames:
#   python lib/extract_frames_ldm_ff++_train.py
#
#   # Train:
#   CUDA_VISIBLE_DEVICES=2 python train.py --cfg ./configs/train_ffpp.cfg
# ─────────────────────────────────────────────────────────────────────────────