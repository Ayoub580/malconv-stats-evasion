# detector/train_malconv.py
"""
MalConv-R training script.

Trains a regularized MalConv reimplementation (MalConv-R) on raw PE bytes
following the official EMBER specification (input_dim=257, padding_char=256),
augmented with BatchNormalization and Dropout for improved generalization.

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
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# ── Paths ─────────────────────────────────────────────────────

BASE_DATA_PATH   = '/home/ayoub/Desktop/ga_overlay/files/balanced_dataset'
TRAIN_DIR        = os.path.join(BASE_DATA_PATH, 'train')
VAL_DIR          = os.path.join(BASE_DATA_PATH, 'val')
TEST_DIR         = os.path.join(BASE_DATA_PATH, 'test')

# ── Hyperparameters ───────────────────────────────────────────

MAX_LEN       = 1024 * 1024   # 1 MB — MalConv receptive field
BATCH_SIZE    = 4             # RTX 4060 8 GB VRAM
EMBEDDING_DIM = 8             # official EMBER spec


# ── Data loading ──────────────────────────────────────────────

def load_pe_file(file_path, label):
    """
    Load a PE file, truncate or zero-pad to MAX_LEN bytes.
    Padding value is 0 (consistent with numpy 'constant' default),
    which is distinct from the dedicated out-of-vocabulary token 256
    used during embedding lookup.
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
                'constant'
            )
        return byte_array, label
    except Exception:
        return np.zeros((MAX_LEN,), dtype=np.int32), label


def load_split(split_dir, shuffle=True):
    """
    Build a tf.data pipeline from a pre-split directory containing
    benign/ and malware/ subdirectories. Undersamples to balance classes.
    """
    benign_dir  = os.path.join(split_dir, 'benign')
    malware_dir = os.path.join(split_dir, 'malware')

    benign_files  = [os.path.join(benign_dir,  f) for f in os.listdir(benign_dir)]
    malware_files = [os.path.join(malware_dir, f) for f in os.listdir(malware_dir)]

    # Undersample to balance classes
    min_count = min(len(benign_files), len(malware_files))
    np.random.shuffle(benign_files)
    np.random.shuffle(malware_files)

    files  = benign_files[:min_count] + malware_files[:min_count]
    labels = [0] * min_count + [1] * min_count

    combined = list(zip(files, labels))
    np.random.shuffle(combined)
    files, labels = zip(*combined)

    ds = tf.data.Dataset.from_tensor_slices((list(files), list(labels)))
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


def get_datasets():
    """
    Load train, val, and test splits from fixed physical directories.
    No random splitting at runtime — partitions are fixed on disk to
    guarantee zero overlap between training and evaluation data.
    """
    print(f"Loading train split from : {TRAIN_DIR}")
    train_ds = load_split(TRAIN_DIR, shuffle=True)

    print(f"Loading val split from   : {VAL_DIR}")
    val_ds = load_split(VAL_DIR, shuffle=False)

    print(f"Loading test split from  : {TEST_DIR}")
    test_ds = load_split(TEST_DIR, shuffle=False)

    return train_ds, val_ds, test_ds


# ── Architecture ──────────────────────────────────────────────

def build_malconv_r():
    """
    MalConv-R: regularized reimplementation of MalConv.

    Follows the official EMBER specification:
      - input_dim = 257  (bytes 0-255 + dedicated padding token 256)
      - embedding_dim = 8
      - gated Conv1D: 128 filters, kernel=512, stride=512
      - GlobalMaxPooling1D

    Augmented with BatchNormalization and Dropout(0.5) for improved
    generalization on our training set. Referred to as MalConv-R in
    the paper to distinguish from the unregularized original.
    """
    inp   = Input(shape=(MAX_LEN,), dtype='int32')

    # Embedding: vocab 257 covers all byte values plus padding token 256
    emb   = layers.Embedding(input_dim=257, output_dim=EMBEDDING_DIM)(inp)

    # Gated temporal convolution (original MalConv design)
    conv1 = layers.Conv1D(128, 512, strides=512, activation='relu')(emb)
    conv2 = layers.Conv1D(128, 512, strides=512, activation='sigmoid')(emb)
    gated = layers.Multiply()([conv1, conv2])

    # Regularization additions (MalConv-R specific)
    x   = layers.BatchNormalization()(gated)
    x   = layers.GlobalMaxPooling1D()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    tf.keras.backend.clear_session()

    train_ds, val_ds, test_ds = get_datasets()

    model = build_malconv_r()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nStarting MalConv-R training ...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stop],
    )

    print("\n" + "=" * 50)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 50)
    results = model.evaluate(test_ds)
    print(f"Test Loss     : {results[0]:.4f}")
    print(f"Test Accuracy : {results[1]:.4f}")
    print("=" * 50)

    model.save('malconv_r.keras')
    print("Model saved → malconv_r.keras")
