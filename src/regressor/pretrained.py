import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Dropout,
    Concatenate,
    Bidirectional,
    LeakyReLU,
)

from src.embedder import BertEmbedder, XLNetEmbedder, RobertaEmbedder


class PretrainedRegressor(Model):
    def __init__(self, config):
        super(PretrainedRegressor, self).__init__()
        self.config = config

        if config["type"].lower() == "bert":
            self.embedding = BertEmbedder(config["embedding"]).get_layer()
        elif config["type"].lower() == "xlnet":
            self.embedding = XLNetEmbedder(config["embedding"]).get_layer()
        elif config["type"].lower() == "roberta":
            self.embedding = RobertaEmbedder(config["embedding"]).get_layer()

        self.recurrent_layers = {"lstm", "bilstm", "gru", "bigru"}

        if config["layer_type"].lower() == "lstm":
            self.text = LSTM(config["text_unit"])
        elif config["layer_type"].lower() == "bilstm":
            self.text = Bidirectional(LSTM(config["text_unit"]))
        elif config["layer_type"].lower() == "gru":
            self.text = GRU(config["text_unit"])
        elif config["layer_type"].lower() == "bigru":
            self.text = Bidirectional(GRU(config["text_unit"]))
        elif config["layer_type"].lower() == "dense":
            self.text = Dense(config["text_unit"])
        else:
            raise ValueError("only support (lstm | bilstm | gru | bigru) layer type")

        if config["enhance_feat"]:
            self.feat = Dense(
                config["feat_unit"],
                activation="relu",
                name="feat",
            )
            self.dropout_feat = Dropout(config["feat_dropout"])

        self.concatenate = Concatenate()
        self.dropout_text = Dropout(config["text_dropout"])

        self.extractor = Dense(config["extractor_unit"], activation="relu")
        self.out = Dense(1, activation="relu", name="regressor")

    def call(self, X, training=None):
        part_taken = (
            "last_hidden_state"
            if self.config["layer_type"] in self.recurrent_layers
            else "pooler_output"
        )

        X_text = self.embedding(X["text"])[part_taken]
        X_text = self.text(X_text)
        X_text = self.dropout_text(X_text, training=training)

        concated = [X_text]

        if self.config["enhance_feat"]:
            X_feat = self.feat(X["features"])
            X_feat = self.dropout_feat(X_feat)
            concated.append(X_feat)

        X_concatenate = self.concatenate(concated)
        X_fin = self.extractor(X_concatenate)
        res = self.out(X_fin)
        return res
