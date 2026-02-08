import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
LUNA_ROOT = "/kaggle/input/luna16"
ANNOTATIONS_PATH = f"{LUNA_ROOT}/annotations.csv"
SUBSETS = [0,1,2,3]  
PATCH_SIZE = 64       
annotations_df = pd.read_csv(ANNOTATIONS_PATH)
print("Annotations:", len(annotations_df))
annotations_df.head()

def load_mhd(path):
    img = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(img)#(Z, Y, X)
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    return volume, origin, spacing
def find_mhd(series_uid):
    for s in SUBSETS:
        files = glob(f"{LUNA_ROOT}/subset{s}/subset{s}/*.mhd")
        for f in files:
            if series_uid in f:
                return f
    return None
def extract_3d_patch(volume, center, size=64):
    D, H, W = volume.shape
    cz, cy, cx = center.astype(int)
    r = size // 2
    #clamp center inside volume
    cz = np.clip(cz, r, D - r)
    cy = np.clip(cy, r, H - r)
    cx = np.clip(cx, r, W - r)
    patch = volume[
        cz-r:cz+r,
        cy-r:cy+r,
        cx-r:cx+r
    ]

    pad_z = max(0, size - patch.shape[0])
    pad_y = max(0, size - patch.shape[1])
    pad_x = max(0, size - patch.shape[2])

    patch = np.pad(
        patch,
        ((0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant"
    )

    return patch[:size, :size, :size]

X = []
y = []
for _, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
    mhd_path = find_mhd(row["seriesuid"])
    if mhd_path is None:
        continue

    volume, origin, spacing = load_mhd(mhd_path)
    center_world = np.array([row["coordZ"], row["coordY"], row["coordX"]])
    center_voxel = np.rint((center_world - origin[::-1]) / spacing[::-1])
    patch = extract_3d_patch(volume, center_voxel, PATCH_SIZE)
    #normalize
    patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)
    X.append(patch[..., np.newaxis])
    y.append(1)  
X = np.array(X)
y = np.array(y)
print("3D samples:", X.shape)

def show_patch(patch):
    mid = patch.shape[0] // 2
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(patch[mid,:,:], cmap='gray')
    plt.title("Axial")

    plt.subplot(1,3,2)
    plt.imshow(patch[:,mid,:], cmap='gray')
    plt.title("Coronal")

    plt.subplot(1,3,3)
    plt.imshow(patch[:,:,mid], cmap='gray')
    plt.title("Sagittal")

    plt.show()

show_patch(X[0][:,:,:,0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def build_3d_cnn(input_shape=(64,64,64,1)):
    model = models.Sequential([
        layers.Conv3D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling3D(2),

        layers.Conv3D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling3D(2),

        layers.Conv3D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling3D(2),

        layers.GlobalAveragePooling3D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_3d_cnn()
model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=4,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('3D CNN Accuracy')
plt.legend(['Train','Val'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('3D CNN Loss')
plt.legend(['Train','Val'])
plt.show()

