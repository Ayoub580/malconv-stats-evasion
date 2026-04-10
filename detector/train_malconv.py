# detector/train_malconv.py
"""
MalConv-R training script.

Trains a regularized MalConv reimplementation (MalConv-R) on raw PE bytes
following the official EMBER specification (input_dim=257, padding_char=256),
augmented with BatchNormalization and Dropout for improved generalization.

This is the exact script used to train the MalConv-R model evaluated in:
  "Practical Adversarial Evasion of MalConv via Model-Scored Overlay Selection"

Dataset layout expected:
    balanced_dataset/
        train/
            benign/
            malware/
        val/
            benign/
            malware/
        test/
            benign/
            malware/

Reference:
    Raff et al., "Malware Detection by Eating a Whole EXE", AAAI 2018.
    Anderson & Roth, "EMBER", arXiv:1804.04637.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks

# ── System ────────────────────────────────────────────────────

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# ── Reproducibility ───────────────────────────────────────────

SEED = 23
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────

BASE_DATA_PATH = 'balanced_dataset'
TRAIN_DIR      = os.path.join(BASE_DATA_PATH, 'train')
VAL_DIR        = os.path.join(BASE_DATA_PATH, 'val')
TEST_DIR       = os.path.join(BASE_DATA_PATH, 'test')

# ── Hyperparameters ───────────────────────────────────────────

MAX_LEN       = 1024 * 1024  # 1 MB — MalConv receptive field
BATCH_SIZE    = 4             
EMBEDDING_DIM = 8             # official EMBER spec
EPOCHS        = 30
MODEL_NAME    = 'malconv_r.keras'



# ── Data loading ──────────────────────────────────────────────

def load_pe_file(file_path, label):
    """
    Read a PE file, convert to int32, and pad to MAX_LEN.

    Padding value is 256 — a dedicated out-of-vocabulary token that is
    distinct from real null bytes (0), matching the official EMBER MalConv
    specification (input_dim=257, padding_char=256).
    """
    path = file_path.numpy().decode('utf-8')
    try:
        with open(path, 'rb') as f:
            raw_bytes = f.read(MAX_LEN)
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int32)
        if len(byte_array) < MAX_LEN:
            byte_array = np.pad(
                byte_array,
                (0, MAX_LEN - len(byte_array)),
                'constant',
                constant_values=256  
            )
        return byte_array, label
    except Exception:
        return np.zeros((MAX_LEN,), dtype=np.int32), label


def get_dataset(base_dir):
    """
    Build a tf.data pipeline from a pre-split directory.

    Loads benign/ and malware/ subdirectories, computes class weights
    to handle imbalance, and returns the dataset and weight dict.
    Shuffling is seeded via the global SEED for reproducibility.
    """
    benign_dir  = os.path.join(base_dir, 'benign')
    malware_dir = os.path.join(base_dir, 'malware')

    benign_files  = [os.path.join(benign_dir,  f) for f in os.listdir(benign_dir)]
    malware_files = [os.path.join(malware_dir, f) for f in os.listdir(malware_dir)]

    files  = benign_files + malware_files
    labels = [0] * len(benign_files) + [1] * len(malware_files)

    # Class weights to handle potential imbalance
    total         = len(labels)
    weight_for_0  = (1 / len(benign_files))  * (total / 2.0)
    weight_for_1  = (1 / len(malware_files)) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.shuffle(len(files), seed=SEED, reshuffle_each_iteration=False)
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
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), class_weights


# ── Architecture ──────────────────────────────────────────────

def build_malconv_r():
    """
    MalConv-R: regularized reimplementation of MalConv.

    Follows the official EMBER specification:
      - input_dim = 257  (bytes 0-255 + dedicated padding token 256)
      - embedding_dim = 8
      - Gated Conv1D: 128 filters, kernel=512, stride=512
      - GlobalMaxPooling1D

    Augmented with BatchNormalization (after GlobalMaxPooling) and
    Dropout(0.5) for improved generalization.
    Optimizer: Adam(lr=1e-4).
    Referred to as MalConv-R in the paper.
    """
    inp = Input(shape=(MAX_LEN,), dtype='int32')

    # Embedding: vocab 257 covers all byte values plus padding token 256
    emb = layers.Embedding(input_dim=257, output_dim=EMBEDDING_DIM)(inp)

    # Gated temporal convolution (original MalConv design)
    conv1 = layers.Conv1D(128, 512, strides=512, activation='relu')(emb)
    conv2 = layers.Conv1D(128, 512, strides=512, activation='sigmoid')(emb)
    gated = layers.Multiply()([conv1, conv2])

    # GlobalMaxPooling then regularization (MalConv-R specific order)
    x   = layers.GlobalMaxPooling1D()(gated)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


# ── Entry point ───────────────────────────────────────────────

def main():
    setup_gpu()
    tf.keras.backend.clear_session()

    print(" Loading datasets...")
    train_ds, train_weights = get_dataset(TRAIN_DIR)
    val_ds,   _             = get_dataset(VAL_DIR)
    test_ds,  _             = get_dataset(TEST_DIR)

    print(" Building MalConv-R...")
    model = build_malconv_r()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"Starting training. Class weights: {train_weights}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=train_weights,
        callbacks=[early_stop],
    )

    print("\n" + "=" * 50)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 50)
    results = model.evaluate(test_ds)
    print(f"Test Loss     : {results[0]:.4f}")
    print(f"Test Accuracy : {results[1]:.4f}")
    print(f"Test AUC      : {results[2]:.4f}")
    print("=" * 50)

    print(f"Saving model → {MODEL_NAME}")
    model.save(MODEL_NAME)
    print("Training complete.")


if __name__ == "__main__":
    main()
