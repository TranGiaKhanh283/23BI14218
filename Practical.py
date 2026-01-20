import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import warnings

warnings.filterwarnings('ignore')
np.random.seed(0)
random_seed = 42


# 23bi14218,vdvndvdnvdvv
train = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header=None)
test = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header=None)

# Balance the dataset
df_balanced = [train[train[187]==0].sample(n=20000, random_state=42)]
df_balanced += [resample(train[train[187]==i], replace=True, n_samples=20000, random_state=random_seed+i) for i in range(1, 5)]
train = pd.concat(df_balanced)

# Split features and labels
X_train, y_train = train.iloc[:, :187], train[187]
X_test, y_test = test.iloc[:, :187], test[187]

# Normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

#  model
model = Sequential([
    Input(shape=(X_train_norm.shape[1], 1)),
    Conv1D(128, 11, activation='relu', padding='Same'),
    MaxPool1D(3, 2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(3, 2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(3, 2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=True))

# Training
model.fit(X_train_norm, y_train, epochs=16, batch_size=32)

# Evaluattionnn
for X, y, name in [(X_test_norm, y_test, 'Test'), (X_train_norm, y_train, 'Train')]:
    y_pred = tf.nn.softmax(model.predict(X)).numpy().argmax(axis=1)
    acc = round((y_pred == y).sum() / y.shape[0] * 100, 2)
    print(f'{name} set accuracy is {acc}%')