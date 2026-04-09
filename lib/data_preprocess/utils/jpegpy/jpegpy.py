# #!/usr/bin/env mdl
# import cv2
# import numpy as np

# from . import _jpegpy


# def jpeg_encode(img: np.array, quality=80):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return _jpegpy.encode(img, quality)


# def jpeg_decode(code: bytes):
#     img = _jpegpy.decode(code)
#     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # vim: ts=4 sw=4 sts=4 expandtab



#!/usr/bin/env python3
import cv2
import numpy as np

try:
    from . import _jpegpy

    def jpeg_encode(img: np.ndarray, quality=80):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return _jpegpy.encode(img, quality)

    def jpeg_decode(code: bytes):
        img = _jpegpy.decode(code)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

except Exception:
    def jpeg_encode(img: np.ndarray, quality=80):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        success, enc = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not success:
            raise RuntimeError("OpenCV jpeg encoding failed")
        return enc.tobytes()

    def jpeg_decode(code: bytes):
        arr = np.frombuffer(code, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("OpenCV jpeg decoding failed")
        return img

# vim: ts=4 sw=4 sts=4 expandtab