!pip install -U ultralytics
import kagglehub
path = kagglehub.dataset_download("anasmohammedtahir/covidqu")

print("Path to dataset files:", path)

!apt-get install tree
!tree -L 5 /root/.cache/kagglehub/datasets/anasmohammedtahir/covidqu/versions/7

import os

for root, dirs, files in os.walk(path):
    print(root, " | files:", len(files))

DATASET_ROOT = "/kaggle/input/covidqu"
INFECTION_ROOT = os.path.join(
    DATASET_ROOT,
    "Infection Segmentation Data",
    "Infection Segmentation Data"
)

YOLO_ROOT = "/content/yolo_covid"

print(INFECTION_ROOT)

for p in [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val", "labels/test"
]:
    os.makedirs(os.path.join(YOLO_ROOT, p), exist_ok=True)


import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil

CLASS_ID = 0   #infection

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        c = c.squeeze()
        if len(c.shape) == 2:
            polys.append(c)
    return polys

def convert_split(split):
    out_img_dir = os.path.join(YOLO_ROOT, "images", split.lower())
    out_lbl_dir = os.path.join(YOLO_ROOT, "labels", split.lower())

    classes = ["COVID-19", "Non-COVID", "Normal"]

    for cls in classes:
        img_dir  = os.path.join(INFECTION_ROOT, split, cls, "images")
        mask_dir = os.path.join(INFECTION_ROOT, split, cls, "infection masks")
        imgs = sorted(glob(img_dir + "/*"))
        masks = sorted(glob(mask_dir + "/*"))

        for img_path, mask_path in tqdm(zip(imgs, masks), total=len(imgs)):
            name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(out_img_dir, name))
            mask = cv2.imread(mask_path, 0)
            h, w = mask.shape
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            polys = mask_to_polygons(mask)

            label_path = os.path.join(
                out_lbl_dir,
                os.path.splitext(name)[0] + ".txt"
            )

            with open(label_path, "w") as f:
                for poly in polys:
                    norm = []
                    for x, y in poly:
                        norm.append(x / w)
                        norm.append(y / h)
                    norm = " ".join([f"{v:.6f}" for v in norm])
                    f.write(f"{CLASS_ID} {norm}\n")
convert_split("Train")
convert_split("Val")
convert_split("Test")

YOLO_ROOT = "/content/yolo_covid"
os.makedirs(YOLO_ROOT, exist_ok=True)
yaml_path = os.path.join(YOLO_ROOT, "data.yaml")

yaml_text = f"""
path: {YOLO_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["infection"]
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text.strip())

print("Created:", yaml_path)
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
model.train(
    data="/content/yolo_covid/data.yaml",
    epochs=20,
    imgsz=512,
    batch=16,
    device=0,
    workers=4
)
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

model = YOLO("/content/runs/segment/train2/weights/best.pt")

test_images = [
    os.path.join(INFECTION_ROOT, "Test", "COVID-19", "images",
                 os.listdir(os.path.join(INFECTION_ROOT, "Test", "COVID-19", "images"))[0]),

    os.path.join(INFECTION_ROOT, "Test", "Non-COVID", "images",
                 os.listdir(os.path.join(INFECTION_ROOT, "Test", "Non-COVID", "images"))[0]),

    os.path.join(INFECTION_ROOT, "Test", "Normal", "images",
                 os.listdir(os.path.join(INFECTION_ROOT, "Test", "Normal", "images"))[0]),
]

results = model(test_images, conf=0.25)

for i, r in enumerate(results):
    img = r.plot()
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(["COVID-19 sample", "Non-COVID sample", "Normal sample"][i])
    plt.axis("off")
    plt.show()

import glob
runs = sorted(glob.glob("runs/segment/train*"))
runs[-1]
import pandas as pd, matplotlib.pyplot as plt
run_dir = sorted(glob.glob("runs/segment/train*"))[-1]
df = pd.read_csv(f"{run_dir}/results.csv")
df.columns

model = YOLO(sorted(glob.glob("runs/segment/train2/weights/best.pt"))[-1])
metrics = model.val(
    data="/content/yolo_covid/data.yaml",
    split="test",
    imgsz=512
)

plt.figure()
plt.plot(df["metrics/precision(B)"], label="Precision")
plt.plot(df["metrics/recall(B)"], label="Recall")
plt.plot(df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Detection metrics")
plt.show()

plt.figure()
plt.plot(df["metrics/precision(M)"], label="Mask Precision")
plt.plot(df["metrics/recall(M)"], label="Mask Recall")
plt.plot(df["metrics/mAP50(M)"], label="Mask mAP@0.5")
plt.plot(df["metrics/mAP50-95(M)"], label="Mask mAP@0.5:0.95")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Segmentation metrics")
plt.show()

plt.figure()
plt.plot(df["train/box_loss"], label="train box loss")
plt.plot(df["val/box_loss"], label="val box loss")
plt.plot(df["train/seg_loss"], label="train seg loss")
plt.plot(df["val/seg_loss"], label="val seg loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curvess")
plt.show()
