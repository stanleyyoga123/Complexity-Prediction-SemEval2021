from transformers import TFBertModel


class BertEmbedder:
    def __init__(self, config):
        self.config = config
        self.embedder = TFBertModel.from_pretrained(config["model_name"])

    def get_layer(self):
        return self.embedder
