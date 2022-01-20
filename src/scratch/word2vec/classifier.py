from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    LSTM,
    Dropout,
    Concatenate,
)


class Classifier(Model):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.embedding = Embedding(**config["embedding"])
        self.lstm_sentence = LSTM(config["lstm_sentence_unit"])
        self.lstm_token = LSTM(config["lstm_token_unit"])
        self.concatenate = Concatenate()
        self.dropout = Dropout(config["dropout_rate"])

        self.dense = Dense(1, activation="relu")

    def call(self, X, training=None):
        X_sentence = self.embedding(X["sentence"])
        X_sentence = self.lstm_sentence(X_sentence)
        X_sentence = self.dropout(X_sentence)

        X_token = self.embedding(X["token"])
        X_token = self.lstm_token(X_token)
        X_token = self.dropout(X_token, training=training)

        X_concatenate = self.concatenate([X_sentence, X_token])
        res = self.dense(X_concatenate)
        return res
