# code_2/CADDM/lib/extract_frames_ldm_ff++.py

#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
from imutils import face_utils


VIDEO_PATH = "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
SAVE_IMGS_PATH = "./test_images"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
DATASETS = {"Original", "FaceSwap", "FaceShifter", "Face2Face", "Deepfakes", "NeuralTextures"}
COMPRESSION = {"raw"}
NUM_FRAMES = 1
IMG_META_DICT = {}


def parse_video_path(dataset, compression):
    # this path setting follows FF++ dataset
    if dataset == "Original":
        dataset_path = f"{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/"
    elif dataset in ["FaceShifter", "Face2Face", "Deepfakes", "FaceSwap", "NeuralTextures"]:
        dataset_path = f"{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/"
    else:
        raise NotImplementedError

    movies_path_list = sorted(glob(dataset_path + "*.mp4"))
    print(f"{len(movies_path_list)} : videos are exist in {dataset}")
    return movies_path_list


def parse_labels(video_path):
    if "original" in video_path:
        return 0
    return 1


def parse_source_save_path(save_path):
    if "original" in save_path:
        return save_path
    else:
        img_meta = save_path.split("/")
        source_target_index = img_meta[-1]
        source_index = source_target_index.split("_")[0]
        manipulation_name = img_meta[-4]
        original_name = "youtube"
        source_save_path = (
            save_path.replace("manipulated_sequences", "original_sequences")
            .replace(manipulation_name, original_name)
            .replace(source_target_index, source_index)
        )
        return source_save_path


def make_dlib_ready(frame):
    """
    Convert OpenCV frame to dlib-compatible uint8 contiguous RGB image.
    Returns None if frame is invalid/unsupported.
    """
    if frame is None:
        return None

    # handle weird dtypes
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # grayscale -> RGB
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # BGRA/RGBA -> RGB
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # standard OpenCV BGR -> RGB
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    else:
        return None

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    return frame


def preprocess_video(video_path, save_path, face_detector, face_predictor):
    label = parse_labels(video_path)
    source_save_path = parse_source_save_path(save_path)
    os.makedirs(save_path, exist_ok=True)

    cap_video = cv2.VideoCapture(video_path)
    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count_video <= 0:
        tqdm.write(f"Invalid video or zero frames: {video_path}")
        cap_video.release()
        return

    frame_idxs = np.linspace(
        0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=int
    )

    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()

        if not ret or frame is None:
            tqdm.write(f"Frame read {cnt_frame} Error! : {os.path.basename(video_path)}")
            continue

        if cnt_frame not in frame_idxs:
            continue

        frame_rgb = make_dlib_ready(frame)
        if frame_rgb is None:
            tqdm.write(
                f"Unsupported image type/shape at frame {cnt_frame} in {os.path.basename(video_path)}"
            )
            continue

        try:
            faces = face_detector(frame_rgb, 1)
        except RuntimeError as e:
            tqdm.write(
                f"dlib failed at frame {cnt_frame} in {os.path.basename(video_path)}: {e}"
            )
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

        # pick the biggest face
        landmarks = np.stack(landmarks, axis=0)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

        video_dict = {
            "landmark": landmarks.tolist(),
            "source_path": f"{source_save_path}/frame_{cnt_frame}",
            "label": label,
        }
        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict

        # save original frame as png
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)

    cap_video.release()


def main():
    os.makedirs(SAVE_IMGS_PATH, exist_ok=True)

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Landmark predictor not found: {PREDICTOR_PATH}")

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    for dataset in DATASETS:
        for comp in COMPRESSION:
            movies_path_list = parse_video_path(dataset, comp)
            n_sample = len(movies_path_list)

            for i in tqdm(range(n_sample), desc=f"Processing {dataset}-{comp}"):
                save_path_per_video = (
                    movies_path_list[i]
                    .replace(VIDEO_PATH, SAVE_IMGS_PATH)
                    .replace(".mp4", "")
                    .replace("/videos", "/frames")
                )

                preprocess_video(
                    movies_path_list[i],
                    save_path_per_video,
                    face_detector,
                    face_predictor,
                )

    with open(f"{SAVE_IMGS_PATH}/ldm.json", "w") as f:
        json.dump(IMG_META_DICT, f)

    print(f"Saved landmark metadata to: {SAVE_IMGS_PATH}/ldm.json")
    print(f"Total processed frames with landmarks: {len(IMG_META_DICT)}")


if __name__ == "__main__":
    main()