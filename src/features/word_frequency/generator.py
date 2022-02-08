import warnings

warnings.filterwarnings("ignore")

import numpy as np
from .loader import Loader


class FrequencyGenerator:
    def __init__(self, list_frequency):
        self.loader = Loader()
        self.dictionary = {}
        for freq_name in list_frequency:
            self.dictionary.update(self.loader.STR_FUNC_MAP[freq_name]())
        self.list_frequency = list(self.dictionary.keys())

    def __call__(self, texts):
        ret = {freq_name: [] for freq_name in self.list_frequency}
        for words in texts:
            words = words.split()
            val = {freq_name: 0 for freq_name in self.list_frequency}
            for word in words:
                for freq_name in self.list_frequency:
                    val[freq_name] += (
                        self.dictionary[freq_name][word]
                        if word in self.dictionary[freq_name]
                        else 0
                    )
            for freq_name in self.list_frequency:
                val[freq_name] /= len(words)
                val[freq_name] = 1 if val[freq_name] == 0 else val[freq_name]
                ret[freq_name].append(val[freq_name])
        for freq_name in self.list_frequency:
            ret[freq_name] = 1 / np.array(ret[freq_name])

        return ret
