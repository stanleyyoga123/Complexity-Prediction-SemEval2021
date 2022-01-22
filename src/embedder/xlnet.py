from transformers import TFXLNetModel


class XLNetEmbedder:
    def __init__(self, config):
        self.config = config
        self.embedder = TFXLNetModel.from_pretrained(config["model_name"])

    def __call__(self, x):
        return self.embedder(x)
