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


# Load data
train = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header=None)
test = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header=None)

# Balance dataset
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

model.summary()


model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=True))

# Train
model.fit(X_train_norm, y_train, epochs=16, batch_size=32)

# Evaluattionnn
for X, y, name in [(X_test_norm, y_test, 'Test'), (X_train_norm, y_train, 'Train')]:
    y_pred = tf.nn.softmax(model.predict(X)).numpy().argmax(axis=1)
    acc = round((y_pred == y).sum() / y.shape[0] * 100, 2)
    print(f'{name} set accuracy is {acc}%')

from sklearn.metrics import confusion_matrix, classification_report
def evaluate_model(X, y, name):
    y_logits = model.predict(X)
    y_pred = tf.nn.softmax(y_logits).numpy().argmax(axis=1)

    # Accuracy
    acc = round((y_pred == y).sum() / y.shape[0] * 100, 2)
    print(f'\n{name} set accuracy: {acc}%')

    # Classification Report
    print(f'\n{name} Classification Report:')
    print(classification_report(y, y_pred, digits=4))
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Pred {i}' for i in range(cm.shape[0])],
        yticklabels=[f'True {i}' for i in range(cm.shape[0])]
    )
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

evaluate_model(X_test_norm, y_test, 'Test')
evaluate_model(X_train_norm, y_train, 'Train')
