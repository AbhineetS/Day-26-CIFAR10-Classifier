#!/usr/bin/env python3
"""
Day 26 — CIFAR-10 Image Classifier
Simple, well-commented training script. Saves model and training plot.
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

MODEL_OUT = "cnn_cifar10_day26.keras"
PLOT_OUT = "training_history_day26.png"
BATCH_SIZE = 64
IMG_SHAPE = (32, 32, 3)
EPOCHS = 12

def build_model(input_shape=IMG_SHAPE, num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0/255)(inputs)

    # Conv block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Conv block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cifar10_cnn")
    return model

def prepare_data(batch_size=BATCH_SIZE):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomRotation(0.08),
        ],
        name="data_augmentation"
    )

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size)
    train_ds = train_ds.map(lambda x,y: (data_augmentation(tf.cast(x, tf.float32), training=True), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def compile_and_train(model, train_ds, val_ds, epochs=EPOCHS):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2)
    return history

def save_artifacts(model, history):
    model.save(MODEL_OUT)
    print(f"💾 Saved model: {MODEL_OUT}")

    # Plot accuracy
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure()
    plt.plot(acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    plt.close()
    print(f"💾 Saved training plot: {PLOT_OUT}")

def main():
    print("📦 Preparing data...")
    train_ds, val_ds = prepare_data()

    print("🧠 Building model...")
    model = build_model()
    model.summary()

    print("🚀 Training model...")
    history = compile_and_train(model, train_ds, val_ds)

    print("🔍 Evaluating...")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Final Accuracy: {acc:.4f}")

    save_artifacts(model, history)
    print("✅ Done.")

if __name__ == '__main__':
    main()
