from transformers import TFXLNetModel


class XLNetEmbedder:
    def __init__(self, config):
        self.config = config
        self.embedder = TFXLNetModel.from_pretrained(config["model_name"])

    def get_layer(self):
        return self.embedder
