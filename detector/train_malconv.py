# detector/train_malconv.py
"""
MalConv training script.
Trains a gated-CNN malware detector on raw PE bytes, evaluates on a
held-out test set, and saves the model as malconv_robust_final.h5.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.default import (
    TRAIN_BENIGN_PATH, TRAIN_MALWARE_PATH,
    TEST_BENIGN_PATH,  TEST_MALWARE_PATH,
    MAX_LEN, BATCH_SIZE, EMBEDDING_DIM,
)

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


# ── Data loading ──────────────────────────────────────────────

def load_pe_file(file_path, label):
    path = file_path.numpy().decode("utf-8")
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read(MAX_LEN)
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int32)
        if len(byte_array) < MAX_LEN:
            byte_array = np.pad(byte_array, (0, MAX_LEN - len(byte_array)), "constant")
        return byte_array, label
    except Exception:
        return np.zeros((MAX_LEN,), dtype=np.int32), label


def create_tf_dataset(files, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((list(files), list(labels)))
    if shuffle:
        ds = ds.shuffle(len(files))
    ds = ds.map(
        lambda f, l: tf.py_function(load_pe_file, [f, l], [tf.int32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        lambda x, y: (
            tf.ensure_shape(x, (MAX_LEN,)),
            tf.ensure_shape(y, ()),
        )
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def get_datasets(val_split=0.2):
    # Training files
    t_benign  = [os.path.join(TRAIN_BENIGN_PATH,  f) for f in os.listdir(TRAIN_BENIGN_PATH)]
    t_malware = [os.path.join(TRAIN_MALWARE_PATH, f) for f in os.listdir(TRAIN_MALWARE_PATH)]

    min_train = min(len(t_benign), len(t_malware))
    np.random.shuffle(t_benign)
    np.random.shuffle(t_malware)
    t_files  = t_benign[:min_train]  + t_malware[:min_train]
    t_labels = [0] * min_train + [1] * min_train

    indices = np.arange(len(t_files))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))

    train_ds = create_tf_dataset(
        np.array(t_files)[indices[:split]],
        np.array(t_labels)[indices[:split]],
    )
    val_ds = create_tf_dataset(
        np.array(t_files)[indices[split:]],
        np.array(t_labels)[indices[split:]],
    )

    # Test files
    test_benign  = [os.path.join(TEST_BENIGN_PATH,  f) for f in os.listdir(TEST_BENIGN_PATH)]
    test_malware = [os.path.join(TEST_MALWARE_PATH, f) for f in os.listdir(TEST_MALWARE_PATH)]

    min_test    = min(len(test_benign), len(test_malware))
    test_files  = test_benign[:min_test]  + test_malware[:min_test]
    test_labels = [0] * min_test + [1] * min_test
    test_ds     = create_tf_dataset(test_files, test_labels, shuffle=False)

    return train_ds, val_ds, test_ds


# ── Architecture ──────────────────────────────────────────────

def build_malconv():
    """
    Gated temporal convolutional network operating on raw PE bytes.
    Follows the original MalConv architecture (Raff et al., 2018).
    """
    inp   = Input(shape=(MAX_LEN,), dtype="int32")
    emb   = layers.Embedding(input_dim=257, output_dim=EMBEDDING_DIM)(inp)
    conv1 = layers.Conv1D(128, 512, strides=512, activation="relu")(emb)
    conv2 = layers.Conv1D(128, 512, strides=512, activation="sigmoid")(emb)
    gated = layers.Multiply()([conv1, conv2])
    x     = layers.BatchNormalization()(gated)
    x     = layers.GlobalMaxPooling1D()(x)
    x     = layers.Dropout(0.5)(x)
    x     = layers.Dense(128, activation="relu")(x)
    x     = layers.Dropout(0.5)(x)
    out   = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    tf.keras.backend.clear_session()

    train_ds, val_ds, test_ds = get_datasets()
    model = build_malconv()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print("Starting training ...")
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop])

    print("\n" + "=" * 40)
    print("EVALUATION ON HELD-OUT TEST SET")
    results = model.evaluate(test_ds)
    print(f"Test Loss     : {results[0]:.4f}")
    print(f"Test Accuracy : {results[1]:.4f}")
    print("=" * 40)

    model.save("malconv_robust_final.h5")
    print("Model saved → malconv_robust_final.h5")
