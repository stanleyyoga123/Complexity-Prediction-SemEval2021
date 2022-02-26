import numpy as np
import os
import pickle
import pandas as pd

from src.constant import Path
from src.loader import ScratchLoader
from src.regressor import ScratchRegressor
from src.embedder import Word2VecEmbedder

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
)

from tqdm import tqdm

config = {
    "mode": "patch",
    "master": {"enhance_feat": True},
    "loader": {
        "tokenizer": {
            "lower": True,
            "split": " ",
            "char_level": 0,
            "oov_token": "<OOV>",
            "num_words": 10000,
        },
        "features": "all",
        "sentence_maxlen": 256,
        "token_maxlen": 2,
    },
    "word2vec": {
        "vector_size": 256,
        "window": 5,
        "min_count": 1,
        "workers": -1,
        "seed": 42,
    },
    "fasttext": {
        "vector_size": 256,
        "window": 3,
        "min_count": 1,
        "workers": -1,
        "seed": 42,
    },
    "regressor": {
        "trainable_embedding": False,
        "layer_type": "bigru",
        "sentence_unit": 64,
        "token_unit": 64,
        "dropout_rate": 0.5,
        "dense_unit": 32,
    },
}


def add_features(data, features):
    ret = data.copy()
    ret["X_train"]["features"] = np.array(loader.train[features])
    ret["X_dev"]["features"] = np.array(loader.dev[features])
    ret["X_test"]["features"] = np.array(loader.test[features])
    return ret


df = pd.read_csv(Path.TRAIN_SINGLE)
bank_feats = list(df.columns[5:])

loader = ScratchLoader({**config["master"], **config["loader"]})
raw_data = loader()

embedder = Word2VecEmbedder({**config["word2vec"]})
embedder.fit(raw_data["train_embedder"])
embedder_config = embedder.get_embedding_config(loader.tokenizer)


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)


callbacks = [
    EarlyStopping(patience=5),
    LearningRateScheduler(scheduler),
]

feature_scores = {}
features = []

while True:
    print(f"Iterations {len(features) + 1}")
    val_loss = {}
    for i, feature in enumerate(bank_feats):
        if feature in features:
            continue

        used_features = [*features, feature]
        print(f"Trying Features {i}: {' '.join(used_features)}")
        data = add_features(raw_data, used_features)
        print(f"Feature Data Shape {data['X_train']['features'].shape}")

        model = ScratchRegressor(
            {**config["master"], **config["regressor"], "embedding": embedder_config}
        )
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-4))

        history = model.fit(
            data["X_train"],
            data["y_train"],
            batch_size=64,
            epochs=50,
            validation_data=(data["X_dev"], data["y_dev"]),
            callbacks=callbacks,
            verbose=0,
        )
        val_loss[feature] = min(history.history["val_loss"])
        print(f"Features {' '.join(used_features)} got {val_loss[feature]}")

    taken_feat, min_ = None, None
    for key, value in val_loss.items():
        if not taken_feat:
            taken_feat = key
            min_ = value
            continue

        if value < min_:
            taken_feat = key
            min_ = value

    features.append(taken_feat)
    print(f"Taken {taken_feat} with {min_} validation loss")
    feature_scores["|".join(features)] = min_
    print()

    pickle.dump(
        feature_scores, open(os.path.join("resources", "feature-selection.pkl"), "wb")
    )

    if len(features) == len(bank_feats):
        break
