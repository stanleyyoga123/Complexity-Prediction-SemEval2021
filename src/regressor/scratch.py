import tensorflow as tf

tf.random.set_seed(42)

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    LSTM,
    GRU,
    Dropout,
    Concatenate,
    Bidirectional,
)


class ScratchRegressor(Model):
    def __init__(self, config):
        super(ScratchRegressor, self).__init__()
        self.config = config
        # self.embedding = Embedding(**config["embedding"])
        # self.embedding = config["embedding"]

        if config["layer_type"].lower() == "lstm":
            self.sentence = LSTM(config["sentence_unit"])
            self.token = LSTM(config["token_unit"])
        elif config["layer_type"].lower() == "bilstm":
            self.sentence = Bidirectional(LSTM(config["sentence_unit"]))
            self.token = Bidirectional(LSTM(config["token_unit"]))
        elif config["layer_type"].lower() == "gru":
            self.sentence = GRU(config["sentence_unit"])
            self.token = GRU(config["token_unit"])
        elif config["layer_type"].lower() == "bigru":
            self.sentence = Bidirectional(GRU(config["sentence_unit"]))
            self.token = Bidirectional(GRU(config["token_unit"]))
        else:
            raise ValueError("only support (lstm | bilstm | gru | bigru) layer type")

        if config["enhance_feat"]:
            self.extractor = Dense(
                config["dense_unit"], activation="relu", name="extractor"
            )

        self.concatenate = Concatenate()
        self.dropout = Dropout(config["dropout_rate"])

        self.dense = Dense(1, activation="relu", name="regressor")

    def call(self, X, training=None):
        X_sentence = self.sentence(X["sentence"])
        X_sentence = self.dropout(X_sentence)

        X_token = self.token(X["token"])
        X_token = self.dropout(X_token, training=training)

        concated = [X_sentence, X_token]

        if self.config["enhance_feat"]:
            X_dense = self.extractor(X["features"])
            X_dense = self.dropout(X_dense)
            concated.append(X_dense)

        X_concatenate = self.concatenate(concated)
        res = self.dense(X_concatenate)
        return res