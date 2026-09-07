"""
convert_yolo_to_classification.py

Converts a YOLOv8-format object detection dataset (images + label .txt files
with bounding boxes) into a classification-ready folder structure:

    dataset/
      train/
        bottle/
        umbrella/
        calculator/
        charger/
        key/
      val/
        bottle/
        umbrella/
        calculator/
        charger/
        key/

It does this by cropping out each labeled bounding box from its image and
saving the crop into the matching class folder. If an image has multiple
objects, each one is saved as a separate cropped photo.

HOW TO USE:
1. Download the dataset from Roboflow in "YOLOv8" format and unzip it.
   You should see folders like:
     train/images/, train/labels/
     valid/images/, valid/labels/
     (there may also be a test/ folder — you can ignore it for now)
   There will also be a data.yaml file listing the class names in order.

2. Put this script in the SAME folder as the unzipped dataset (so it can
   see train/, valid/, and data.yaml next to it).

3. Run: python convert_yolo_to_classification.py

4. This creates a new "dataset/" folder next to it, already organized
   for the classifier notebook. Just point DATA_DIR in the notebook to
   this new "dataset" folder (or move/rename it to replace the old one).
"""

import os
import yaml
from PIL import Image

# ---- CONFIG ----
# Some Roboflow exports name the validation split "valid", others "test".
# This automatically picks whichever one actually exists.
SOURCE_TRAIN_IMAGES = "train/images"
SOURCE_TRAIN_LABELS = "train/labels"

if os.path.isdir("valid/images"):
    SOURCE_VAL_IMAGES = "valid/images"
    SOURCE_VAL_LABELS = "valid/labels"
elif os.path.isdir("test/images"):
    SOURCE_VAL_IMAGES = "test/images"
    SOURCE_VAL_LABELS = "test/labels"
else:
    SOURCE_VAL_IMAGES = "valid/images"  # will just be skipped with a warning
    SOURCE_VAL_LABELS = "valid/labels"

DATA_YAML = "data.yaml"

OUTPUT_DIR = "dataset"
# ------------------------------------------------------------


def load_class_names():
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def yolo_box_to_pixels(x_center, y_center, w, h, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def process_split(images_dir, labels_dir, split_name, class_names):
    out_split_dir = os.path.join(OUTPUT_DIR, split_name)
    for cname in class_names:
        os.makedirs(os.path.join(out_split_dir, cname), exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"Skipping {split_name}: '{images_dir}' not found.")
        return 0

    count = 0
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not open {image_path}: {e}")
            continue

        img_w, img_h = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])

            if class_id >= len(class_names):
                continue

            class_name = class_names[class_id]
            x1, y1, x2, y2 = yolo_box_to_pixels(x_center, y_center, w, h, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            out_name = f"{os.path.splitext(fname)[0]}_{i}.jpg"
            out_path = os.path.join(out_split_dir, class_name, out_name)
            crop.save(out_path, "JPEG")
            count += 1

    return count


def main():
    class_names = load_class_names()
    print("Classes found:", class_names)

    train_count = process_split(SOURCE_TRAIN_IMAGES, SOURCE_TRAIN_LABELS, "train", class_names)
    val_count = process_split(SOURCE_VAL_IMAGES, SOURCE_VAL_LABELS, "val", class_names)

    print(f"\nDone. Cropped {train_count} training images and {val_count} validation images.")
    print(f"Output saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nNext step: point your classifier notebook's DATA_DIR to this 'dataset' folder.")


if __name__ == "__main__":
    main()
