from transformers import TFRobertaModel

class RobertaEmbedder:
    def __init__(self, config):
        self.config = config
        self.embedder = TFRobertaModel.from_pretrained(config["model_name"])

    def get_layer(self):
        return self.embedder
