# #!/usr/bin/env python3
# """
# lib/diagnose.py  — run this FIRST to find the exact problem.

# cd /scratch/sahil/projects/img_deepfake/code_2/img_fake-cadm
# python lib/diagnose.py
# """
# import cv2
# import numpy as np
# import subprocess
# import tempfile
# import os
# import sys
# import dlib

# VIDEO_PATH = "/scratch/sahil/projects/img_deepfake/datasets/ffpp"
# TEST_VIDEO = f"{VIDEO_PATH}/original_sequences/youtube/raw/videos/000.mp4"
# PREDICTOR  = "./lib/shape_predictor_81_face_landmarks.dat"

# print("=" * 60)
# print(f"Python  : {sys.version}")
# print(f"OpenCV  : {cv2.__version__}")
# print(f"dlib    : {dlib.__version__}")
# print(f"numpy   : {np.__version__}")
# print("=" * 60)

# # ── 1. Read via OpenCV VideoCapture ─────────────────────────────────────────
# print("\n[TEST 1] OpenCV VideoCapture")
# cap = cv2.VideoCapture(TEST_VIDEO)
# ret, frame_cv = cap.read()
# cap.release()
# if ret and frame_cv is not None:
#     print(f"  dtype={frame_cv.dtype}  shape={frame_cv.shape}  "
#           f"min={frame_cv.min()}  max={frame_cv.max()}")
# else:
#     print("  FAILED to read frame")
#     frame_cv = None

# # ── 2. Extract frame 0 via ffmpeg ────────────────────────────────────────────
# print("\n[TEST 2] ffmpeg frame extraction")
# with tempfile.TemporaryDirectory() as tmpdir:
#     out_png = os.path.join(tmpdir, "frame.png")
#     cmd = [
#         "ffmpeg", "-loglevel", "error",
#         "-i", TEST_VIDEO,
#         "-vf", "select=eq(n\\,0)",
#         "-vframes", "1",
#         out_png, "-y"
#     ]
#     r = subprocess.run(cmd, capture_output=True)
#     if r.returncode == 0 and os.path.exists(out_png):
#         frame_ffmpeg = cv2.imread(out_png, cv2.IMREAD_COLOR)
#         print(f"  dtype={frame_ffmpeg.dtype}  shape={frame_ffmpeg.shape}  "
#               f"min={frame_ffmpeg.min()}  max={frame_ffmpeg.max()}")
#     else:
#         print(f"  ffmpeg FAILED: {r.stderr.decode()[:200]}")
#         frame_ffmpeg = None

# # ── 3. Try every possible way to call dlib ───────────────────────────────────
# print("\n[TEST 3] dlib face_detector — trying every image format")
# detector = dlib.get_frontal_face_detector()

# def try_dlib(name, img):
#     try:
#         result = detector(img, 1)
#         print(f"  [{name}]  ✓ OK — {len(result)} face(s) detected")
#         return True
#     except RuntimeError as e:
#         print(f"  [{name}]  ✗ RuntimeError: {e}")
#         return False
#     except Exception as e:
#         print(f"  [{name}]  ✗ {type(e).__name__}: {e}")
#         return False

# for src_name, frame in [("opencv", frame_cv), ("ffmpeg", frame_ffmpeg)]:
#     if frame is None:
#         continue
#     print(f"\n  --- source: {src_name} ---")

#     # BGR as-is
#     try_dlib(f"{src_name}_bgr_uint8", frame)

#     # BGR→RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
#     try_dlib(f"{src_name}_rgb_uint8", rgb)

#     # Grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = np.ascontiguousarray(gray, dtype=np.uint8)
#     try_dlib(f"{src_name}_gray_uint8", gray)

#     # dlib.load_rgb_image path (avoids numpy entirely)
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
#         tmp = f.name
#     cv2.imwrite(tmp, frame)
#     try:
#         img_dlib = dlib.load_rgb_image(tmp)
#         try_dlib(f"{src_name}_dlib_load_rgb", img_dlib)
#     except Exception as e:
#         print(f"  [{src_name}_dlib_load_rgb]  load failed: {e}")
#     os.unlink(tmp)

# # ── 4. Check video codec info ─────────────────────────────────────────────────
# print("\n[TEST 4] Video codec info")
# try:
#     cmd = [
#         "ffprobe", "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries",
#         "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate",
#         "-of", "flat",
#         TEST_VIDEO
#     ]
#     out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
#     print(f"  {out.strip()}")
# except Exception as e:
#     print(f"  ffprobe failed: {e}")

# print("\n" + "=" * 60)
# print("DONE — paste this output in the chat")
# print("=" * 60)



# ==============================================================================
#!/usr/bin/env python3
# """
# lib/verify2.py — tests the REAL fix: dlib.load_rgb_image() instead of numpy.

#   python lib/verify2.py
# """
# import subprocess, shutil, tempfile, os, sys, dlib
# from imutils import face_utils

# VIDEO = "/scratch/sahil/projects/img_deepfake/datasets/ffpp/original_sequences/youtube/raw/videos/000.mp4"
# PRED  = "./lib/shape_predictor_81_face_landmarks.dat"

# print("ffmpeg:", shutil.which("ffmpeg"))
# detector  = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(PRED)

# with tempfile.TemporaryDirectory() as d:
#     for idx in [0, 50, 100, 150, 200, 300]:
#         png = os.path.join(d, f"f{idx}.png")
#         cmd = ["ffmpeg", "-loglevel", "error", "-i", VIDEO,
#                "-vf", f"select=eq(n\\,{idx}),format=rgb24",
#                "-vframes", "1", "-f", "image2", png, "-y"]
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         if not os.path.exists(png) or os.path.getsize(png) < 100:
#             print(f"Frame {idx}: ffmpeg failed")
#             continue

#         # THE FIX: load directly with dlib, no numpy involved
#         try:
#             img = dlib.load_rgb_image(png)
#             print(f"Frame {idx}: dlib.load_rgb_image OK — type={type(img)}")
#             faces = detector(img, 1)
#             print(f"  → {len(faces)} face(s) detected")
#             if faces:
#                 lm = face_utils.shape_to_np(predictor(img, faces[0]))
#                 print(f"  → landmark shape: {lm.shape}")
#                 print(f"\n✓ WORKS! Now run:")
#                 print(f"  python lib/extract_frames_ldm_ff++_train.py")
#                 sys.exit(0)
#         except Exception as e:
#             print(f"Frame {idx}: dlib.load_rgb_image failed: {e}")

# print("\nNo face found — but if load_rgb_image didn't error, the fix works.")
# print("Run: python lib/extract_frames_ldm_ff++_train.py")




# ==============================================================================
#!/usr/bin/env python3
"""
lib/fix_dlib.py
───────────────
Run this ONCE to fix the dlib/numpy incompatibility.

The problem:
  dlib 19.24.0 was compiled against numpy 1.x.
  You have numpy 2.2.6.
  dlib's numpy_image.h uses old ABI → rejects ALL images.

The fix:
  Reinstall dlib from source (pip builds it against your current numpy).

Run:
  python lib/fix_dlib.py

Then verify:
  python lib/fix_dlib.py --verify

Then run extraction:
  python lib/extract_frames_ldm_ff++_train.py
"""

import sys
import subprocess
import argparse


def verify():
    import numpy as np
    import dlib
    import tempfile, os, cv2

    print(f"numpy : {np.__version__}")
    print(f"dlib  : {dlib.__version__}")

    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 200

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    cv2.imwrite(tmp, img)

    try:
        loaded = dlib.load_rgb_image(tmp)
        detector = dlib.get_frontal_face_detector()
        _ = detector(loaded, 1)
        print("✓ dlib works correctly with current numpy!")
        print("\nNow run:")
        print("  python lib/extract_frames_ldm_ff++_train.py")
        return True
    except RuntimeError as e:
        print(f"✗ dlib still broken: {e}")
        print("  Try running this script again or reinstall manually.")
        return False
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Just verify dlib works, don't reinstall")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    print("=" * 60)
    print("Fixing dlib/numpy incompatibility")
    print("=" * 60)

    pip = sys.executable.replace("python", "pip")
    pip_cmd = [sys.executable, "-m", "pip"]

    # Step 1: uninstall current broken dlib
    print("\n[1/3] Uninstalling current dlib...")
    subprocess.run(pip_cmd + ["uninstall", "dlib", "-y"],
                   check=False)

    # Step 2: install cmake if needed (required to build dlib from source)
    print("\n[2/3] Ensuring cmake is available...")
    subprocess.run(pip_cmd + ["install", "cmake", "--quiet"],
                   check=False)

    # Step 3: reinstall dlib from source (builds against current numpy)
    print("\n[3/3] Installing dlib from source (this takes 3-5 minutes)...")
    print("      Building against numpy", end="")
    import numpy as np
    print(f" {np.__version__}...")

    result = subprocess.run(
        pip_cmd + ["install", "dlib", "--no-binary", "dlib", "-v"],
        capture_output=False
    )

    if result.returncode != 0:
        print("\n✗ pip install failed. Try manually:")
        print("  conda install -c conda-forge dlib")
        print("  # or:")
        print("  pip install dlib --no-binary dlib")
        sys.exit(1)

    print("\n[verify] Testing dlib...")
    if verify():
        print("\n✓ All done! Run:")
        print("  python lib/extract_frames_ldm_ff++_train.py")
    else:
        print("\nIf still broken, try conda:")
        print("  conda install -c conda-forge dlib")


if __name__ == "__main__":
    main()