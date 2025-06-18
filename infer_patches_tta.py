from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model
model = YOLO("best.pt")

# TTA helper functions
def horizontal_flip(img):
    return cv2.flip(img, 1)

def vertical_flip(img):
    return cv2.flip(img, 0)

def rotate_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def reverse_transform(pred, transform):
    if transform == "hflip":
        pred.masks.data = pred.masks.data.flip(-1)
    elif transform == "vflip":
        pred.masks.data = pred.masks.data.flip(-2)
    elif transform == "rot90":
        pred.masks.data = pred.masks.data.permute(0, 1, 3, 2).flip(-2)
    return pred

# Run TTA inference
def infer_with_tta(image_path):
    img = cv2.imread(image_path)

    # Original
    results = [model.predict(img, imgsz=640, conf=0.5)[0]]

    # Horizontal Flip
    h_img = horizontal_flip(img)
    h_pred = model.predict(h_img, imgsz=640, conf=0.5)[0]
    results.append(reverse_transform(h_pred, "hflip"))

    # Vertical Flip
    v_img = vertical_flip(img)
    v_pred = model.predict(v_img, imgsz=640, conf=0.5)[0]
    results.append(reverse_transform(v_pred, "vflip"))

    # Rotate 90
    r_img = rotate_90(img)
    r_pred = model.predict(r_img, imgsz=640, conf=0.5)[0]
    results.append(reverse_transform(r_pred, "rot90"))

    # Combine masks (simple average or union logic can go here)
    final_mask = None
    for res in results:
        if res.masks is not None:
            mask = res.masks.data[0].cpu().numpy()
            final_mask = mask if final_mask is None else np.maximum(final_mask, mask)

    if final_mask is not None:
        final_mask = (final_mask * 255).astype(np.uint8)
        out_path = os.path.join("patched_outputs", os.path.basename(image_path))
        os.makedirs("patched_outputs", exist_ok=True)
        cv2.imwrite(out_path, final_mask)
        print(f"✅ Saved TTA mask to {out_path}")
    else:
        print(f"❌ No mask found for {image_path}")

# Example run
if __name__ == "__main__":
    test_folder = "test_images"
    for img_file in os.listdir(test_folder):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            infer_with_tta(os.path.join(test_folder, img_file))
