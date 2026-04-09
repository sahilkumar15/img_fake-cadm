#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
from imutils import face_utils


VIDEO_PATH = "/scratch/sahil/projects/img_deepfake/datasets/celebdf/Celeb-DF_v2"
SAVE_IMGS_PATH = "./test_images_celebdf"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
NUM_FRAMES = 1
IMG_META_DICT = {}


def get_video_lists():
    real_1 = sorted(glob(f"{VIDEO_PATH}/Celeb-real/*.mp4"))
    real_2 = sorted(glob(f"{VIDEO_PATH}/YouTube-real/*.mp4"))
    fake = sorted(glob(f"{VIDEO_PATH}/Celeb-synthesis/*.mp4"))

    print(f"{len(real_1)} videos in Celeb-real")
    print(f"{len(real_2)} videos in YouTube-real")
    print(f"{len(fake)} videos in Celeb-synthesis")

    return real_1, real_2, fake


def parse_label(video_path):
    if "Celeb-synthesis" in video_path:
        return 1
    return 0


def get_save_path(video_path):
    rel = os.path.relpath(video_path, VIDEO_PATH)
    save_path = os.path.join(SAVE_IMGS_PATH, rel)
    save_path = save_path.replace(".mp4", "")
    return save_path


def get_source_path(save_path, label):
    # Celeb-DF does not have FF++-style paired source paths exposed in filenames.
    # For test-time compatibility, point source_path to itself.
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


def preprocess_video(video_path, save_path, face_detector, face_predictor):
    label = parse_label(video_path)
    source_save_path = get_source_path(save_path, label)
    os.makedirs(save_path, exist_ok=True)

    cap_video = cv2.VideoCapture(video_path)
    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count_video <= 0:
        tqdm.write(f"Invalid video or zero frames: {video_path}")
        cap_video.release()
        return

    frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=int)

    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()

        if not ret or frame is None:
            tqdm.write(f"Frame read {cnt_frame} Error! : {os.path.basename(video_path)}")
            continue

        if cnt_frame not in frame_idxs:
            continue

        frame_rgb = make_dlib_ready(frame)
        if frame_rgb is None:
            tqdm.write(f"Unsupported frame type in {os.path.basename(video_path)}")
            continue

        try:
            faces = face_detector(frame_rgb, 1)
        except RuntimeError as e:
            tqdm.write(f"dlib failed at frame {cnt_frame} in {os.path.basename(video_path)}: {e}")
            continue

        if len(faces) == 0:
            tqdm.write(f"No faces in {cnt_frame}:{os.path.basename(video_path)}")
            continue

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

        key = f"{save_path}/frame_{cnt_frame}"
        IMG_META_DICT[key] = {
            "landmark": landmarks.tolist(),
            "source_path": f"{source_save_path}/frame_{cnt_frame}",
            "label": label,
        }

        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)

    cap_video.release()


def main():
    os.makedirs(SAVE_IMGS_PATH, exist_ok=True)

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Landmark predictor not found: {PREDICTOR_PATH}")

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    real_1, real_2, fake = get_video_lists()
    all_videos = real_1 + real_2 + fake

    print(f"Total videos: {len(all_videos)}")

    for video_path in tqdm(all_videos, desc="Processing Celeb-DF"):
        save_path = get_save_path(video_path)
        preprocess_video(video_path, save_path, face_detector, face_predictor)

    with open(f"{SAVE_IMGS_PATH}/ldm.json", "w") as f:
        json.dump(IMG_META_DICT, f)

    print(f"Saved landmark metadata to: {SAVE_IMGS_PATH}/ldm.json")
    print(f"Total processed frames with landmarks: {len(IMG_META_DICT)}")


if __name__ == "__main__":
    main()
    
# python lib/extract_frames_ldm_celebdf.py