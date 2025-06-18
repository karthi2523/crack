# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# import warnings

# # ─── CONFIG ────────────────────────────────────────────────────────────────
# MODEL_PATH   = "best.pt"
# # IMAGE_FOLDER = "test_images"
# VIDEO_FILE   = "test_video.mp4"
# OUTPUT_DIR   = "patched_outputs"
# PATCH_SIZE   = 512
# OVERLAP      = 128
# CONF_THRESH  = 0.2
# IMG_SIZE     = PATCH_SIZE
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# warnings.filterwarnings("ignore")
# # ────────────────────────────────────────────────────────────────────────────

# # Load model
# model = YOLO(MODEL_PATH)

# def infer_patches(frame, patch_size=PATCH_SIZE, overlap=OVERLAP, conf=CONF_THRESH):
#     """Patch-wise inference to catch small cracks with overlap."""
#     h, w = frame.shape[:2]
#     stride = patch_size - overlap
#     canvas = np.zeros((h, w, 3), dtype=np.uint8)

#     for y in range(0, h, stride):
#         for x in range(0, w, stride):
#             x2, y2 = min(x + patch_size, w), min(y + patch_size, h)
#             patch = frame[y:y2, x:x2]

#             results = model.predict(
#                 source=patch,
#                 conf=conf,
#                 imgsz=patch_size,
#                 verbose=False  # suppress logs
#             )
#             rendered = results[0].plot()
#             canvas[y:y2, x:x2] = rendered

#     return canvas

# # ─── IMAGE INFERENCE ───────────────────────────────────────────────────────
# # print("🔍 Running patch-based inference on images...")

# # for fname in os.listdir(IMAGE_FOLDER):
# #     if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
# #         continue
# #     img_path = os.path.join(IMAGE_FOLDER, fname)
# #     frame    = cv2.imread(img_path)
# #     output   = infer_patches(frame)
# #     out_path = os.path.join(OUTPUT_DIR, f"patched_{fname}")
# #     cv2.imwrite(out_path, output)
# #     print(f"  ▶ Saved: {out_path}")

# # ─── VIDEO INFERENCE ───────────────────────────────────────────────────────
# print("\n🎥 Running patch-based inference on video...")

# cap = cv2.VideoCapture(VIDEO_FILE)
# w, h = int(cap.get(3)), int(cap.get(4))
# fps  = cap.get(cv2.CAP_PROP_FPS)
# video_out_path = os.path.join(OUTPUT_DIR, "patched_video.mp4")

# out = cv2.VideoWriter(
#     video_out_path,
#     cv2.VideoWriter_fourcc(*'mp4v'),
#     fps,
#     (w, h)
# )

# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     patched = infer_patches(frame)
#     out.write(patched)
#     frame_count += 1
#     if frame_count % 10 == 0:
#         print(f"  ▶ Processed {frame_count} frames...", end="\r")

# cap.release()
# out.release()

# print("\n✅ Video saved as:", video_out_path)
# print("📁 All outputs saved in:", OUTPUT_DIR)

from ultralytics import YOLO
import cv2
import numpy as np
import os
import warnings

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"
IMAGE_FOLDER = "test_images"
VIDEO_FILE   = "test_video.mp4"
OUTPUT_DIR   = "patched_outputs"
PATCH_SIZE   = 512
OVERLAP      = 128
CONF_THRESH  = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

def infer_patches(frame, patch_size=PATCH_SIZE, overlap=OVERLAP, conf=CONF_THRESH):
    h, w = frame.shape[:2]
    stride = patch_size - overlap
    canvas = frame.copy()

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2, y2 = min(x + patch_size, w), min(y + patch_size, h)
            patch = frame[y:y2, x:x2]

            results = model(patch, conf=conf, imgsz=patch_size, verbose=False)
            r = results[0]

            # Check if masks exist
            if r.masks and r.masks.data is not None:
                masks = r.masks.data.cpu().numpy()

                for mask in masks:
                    # Resize mask to fit current patch size
                    mask_resized = cv2.resize(mask.astype(np.uint8), (x2 - x, y2 - y)) * 255
                    
                    # Blue colored mask (BGR: Blue, 0, 0)
                    blue_mask = cv2.merge([
                        mask_resized,           # Blue
                        np.zeros_like(mask_resized),  # Green
                        np.zeros_like(mask_resized)   # Red
                    ])

                    # Blend mask on top of canvas
                    roi = canvas[y:y2, x:x2]
                    blended = cv2.addWeighted(roi, 1.0, blue_mask, 0.5, 0)
                    canvas[y:y2, x:x2] = blended

    return canvas

# ─── IMAGE INFERENCE ───────────────────────────────────────────────────────
print("🖼 Running patch-based inference on images...")

for fname in os.listdir(IMAGE_FOLDER):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(IMAGE_FOLDER, fname)
    frame    = cv2.imread(img_path)
    if frame is None:
        continue
    output   = infer_patches(frame)
    out_name = os.path.join(OUTPUT_DIR, f"crackmask_{fname}")
    cv2.imwrite(out_name, output)
    print(f"  ▶ Saved: {out_name}")

# ─── VIDEO INFERENCE ───────────────────────────────────────────────────────
# # print("\n🎥 Running patch-based inference on video...")

# # cap = cv2.VideoCapture(VIDEO_FILE)
# # w, h = int(cap.get(3)), int(cap.get(4))
# # fps  = cap.get(cv2.CAP_PROP_FPS)
# # video_out_path = os.path.join(OUTPUT_DIR, "crackmask_video.mp4")

# # out = cv2.VideoWriter(
# #     video_out_path,
# #     cv2.VideoWriter_fourcc(*'mp4v'),
# #     fps,
# #     (w, h)
# # )

# # frame_count = 0
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #     cracked = infer_patches(frame)
# #     out.write(cracked)
# #     frame_count += 1
# #     if frame_count % 10 == 0:
# #         print(f"  ▶ Processed {frame_count} frames...", end="\r")

# cap.release()
# out.release()

print("\n✅ Inference complete!")
print("📁 Crack-only images and video saved in:", OUTPUT_DIR)
