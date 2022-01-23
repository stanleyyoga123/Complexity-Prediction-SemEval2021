import pickle
from src.regressor import PretrainedRegressor
from src.embedder import BertEmbedder

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


config = {
    "embedding": {"model_name": "bert-base-uncased"},
    "type": "bert",
    "layer_type": "bigru",
    "sentence_unit": 64,
    "token_unit": 64,
    "dropout_rate": 0.2,
}

data = pickle.load(open("file.pkl", "rb"))

model = PretrainedRegressor(config)
# out = model(data["X_train"])

# embedder = BertEmbedder(config["embedding"])
# out = embedder(data["X_test"]["token"])

model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-5),
)


model.fit(
    data["X_train"],
    data["y_train"],
    batch_size=64,
    epochs=10,
    validation_data=(data["X_dev"], data["y_dev"]),
)
