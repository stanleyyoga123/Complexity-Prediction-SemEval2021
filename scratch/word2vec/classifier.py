from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    LSTM,
    GRU,
    Dropout,
    Concatenate,
    Bidirectional
)


class Classifier(Model):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.embedding = Embedding(**config["embedding"])

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
        
        self.concatenate = Concatenate()
        self.dropout = Dropout(config["dropout_rate"])

        self.dense = Dense(1, activation="relu")

    def call(self, X, training=None):
        X_sentence = self.embedding(X["sentence"])
        X_sentence = self.sentence(X_sentence)
        X_sentence = self.dropout(X_sentence)

        X_token = self.embedding(X["token"])
        X_token = self.token(X_token)
        X_token = self.dropout(X_token, training=training)

        X_concatenate = self.concatenate([X_sentence, X_token])
        res = self.dense(X_concatenate)
        return res
