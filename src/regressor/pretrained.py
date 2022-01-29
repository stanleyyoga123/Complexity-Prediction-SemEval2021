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
)

from src.embedder import BertEmbedder, XLNetEmbedder


class PretrainedRegressor(Model):
    def __init__(self, config):
        super(PretrainedRegressor, self).__init__()
        self.config = config
        
        if config["type"].lower() == "bert":
            self.embedding = BertEmbedder(config["embedding"]).get_layer()
        elif config["type"].lower() == "xlnet":
            self.embedding = XLNetEmbedder(config["embedding"]).get_layer()
            
        self.recurrent_layers = {"lstm", "bilstm", "gru", "bigru"}

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
        elif config["layer_type"].lower() == "dense":
            self.sentence = Dense(config["sentence_unit"])
            self.token = Dense(config["token_unit"])
        else:
            raise ValueError("only support (lstm | bilstm | gru | bigru) layer type")

        if config["enhance_feat"]:
            self.extractor = Dense(config["dense_unit"])

        self.concatenate = Concatenate()
        self.dropout = Dropout(config["dropout_rate"])

        self.out = Dense(1, activation="relu")

    def call(self, X, training=None):
        part_taken = (
            "last_hidden_state"
            if self.config["layer_type"] in self.recurrent_layers
            else "pooler_output"
        )

        X_sentence = self.embedding(X["sentence"])[part_taken]
        X_sentence = self.sentence(X_sentence)
        X_sentence = self.dropout(X_sentence)

        X_token = self.embedding(X["token"])[part_taken]
        X_token = self.token(X_token)
        X_token = self.dropout(X_token, training=training)

        concated = [X_sentence, X_token]

        if self.config["enhance_feat"]:
            X_dense = self.extractor(X["features"])
            X_dense = self.dropout(X_dense)
            concated.append(X_dense)

        X_concatenate = self.concatenate(concated)
        res = self.out(X_concatenate)
        return res
