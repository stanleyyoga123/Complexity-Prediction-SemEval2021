import numpy as np
from .word_features import FeatureGenerator
from .word_frequency import FrequencyGenerator


class Generator:
    def __init__(self, config):
        feature_names = config["feature_names"].split("|")
        list_frequency = config["list_frequency"].split("|")
        self.feature_generator = FeatureGenerator(feature_names)
        self.frequency_generator = FrequencyGenerator(list_frequency)

    def __call__(self, texts):
        features = self.feature_generator(texts)
        frequencies = self.frequency_generator(texts)
        new_feats = np.array(list({**features, **frequencies}.values())).T
        return new_feats
