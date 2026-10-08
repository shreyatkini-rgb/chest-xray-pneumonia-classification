"""
Chest X-Ray Pneumonia Classification
1) Trains a small CNN from scratch.
2) Trains a MobileNetV2 transfer-learning model.
3) Evaluates the final model and saves plots/metrics.

Expected dataset structure:
data/chest_xray/
    train/NORMAL/*.jpeg
    train/PNEUMONIA/*.jpeg
    test/NORMAL/*.jpeg
    test/PNEUMONIA/*.jpeg
    val/NORMAL/*.jpeg
    val/PNEUMONIA/*.jpeg

The original validation folder is very small, so this script creates
its own validation split from TRAIN and keeps TEST untouched.
"""

import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="training",
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="validation",
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE)

def make_class_weights(data_dir):
    normal = len(os.listdir(os.path.join(data_dir, "train", "NORMAL")))
    pneumonia = len(os.listdir(os.path.join(data_dir, "train", "PNEUMONIA")))
    total = normal + pneumonia
    return {
        0: total / (2.0 * normal),
        1: total / (2.0 * pneumonia)
    }

def plot_history(history, name):
    os.makedirs("outputs", exist_ok=True)
    h = history.history

    plt.figure()
    plt.plot(h["accuracy"], label="Training Accuracy")
    plt.plot(h["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{name.lower()}_accuracy.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(h["loss"], label="Training Loss")
    plt.plot(h["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{name.lower()}_loss.png", dpi=150)
    plt.close()

def build_custom_cnn():
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
    ], name="augmentation")

    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        data_aug,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model

def build_mobilenet():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.06)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model, base

def evaluate(model, test_ds):
    y_true, y_prob = [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().ravel().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0
    )

    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== TEST RESULTS ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    ))
    print(f"ROC-AUC: {auc:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    with open("outputs/test_metrics.json", "w") as f:
        json.dump({
            "accuracy": float((y_true == y_pred).mean()),
            "roc_auc": float(auc),
            "normal_precision": report["NORMAL"]["precision"],
            "normal_recall": report["NORMAL"]["recall"],
            "pneumonia_precision": report["PNEUMONIA"]["precision"],
            "pneumonia_recall": report["PNEUMONIA"]["recall"],
            "pneumonia_f1": report["PNEUMONIA"]["f1-score"]
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_data(args.data)
    class_weights = make_class_weights(args.data)
    print("Class weights:", class_weights)

    # -------- Model 1: Custom CNN --------
    print("\nTraining custom CNN...")
    cnn = build_custom_cnn()
    cnn.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "models/custom_cnn.keras", monitor="val_loss",
            save_best_only=True
        )
    ]

    cnn_history = cnn.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    plot_history(cnn_history, "custom_cnn")

    # -------- Model 2: Transfer Learning --------
    print("\nTraining MobileNetV2 transfer-learning model...")
    model, base = build_mobilenet()
    model.summary()

    callbacks2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "models/xray_mobilenetv2.keras", monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks2
    )
    plot_history(history, "mobilenetv2")

    # Optional light fine-tuning
    print("\nFine-tuning last MobileNetV2 layers...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    fine_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "models/xray_mobilenetv2.keras", monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        class_weight=class_weights,
        callbacks=fine_callbacks
    )

    print("\nEvaluating final model...")
    evaluate(model, test_ds)

    print("\nDONE.")
    print("Model: models/xray_mobilenetv2.keras")
    print("Plots/metrics: outputs/")

if __name__ == "__main__":
    main()
