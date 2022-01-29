import pandas as pd
from src.constant import Path


class Loader:
    """
    Return format
    {word: freq}
    """

    def __init__(self):
        self.STR_FUNC_MAP = {
            "google": Loader.google_freq,
            "wikipedia": Loader.wikipedia_freq,
            "subtlex-us": Loader.subtlex_us_freq,
            "subtlex-uk": Loader.subtlex_uk_freq,
        }

    @staticmethod
    def google_freq():
        df = pd.read_csv(Path.GOOGLE_FREQ, sep="\t")
        df.columns = [" ".join(word.split()) for word in df.columns]
        for col in df.columns[:2]:
            df[col] = df[col].astype(str)

        words = [word.split()[-1] for word in df["#RANKING WORD"]]
        freqs = [int(freq.replace(",", "")) for freq in df["COUNT"]]
        dictionary = {word: freq for word, freq in zip(words, freqs)}
        return {"google-freq": dictionary}

    @staticmethod
    def wikipedia_freq():
        df = pd.read_csv(Path.WIKIPEDIA_FREQ, header=None, sep=" ")
        word_freq = {word: word_freq for word, word_freq in zip(df[0], df[2])}
        doc_freq = {word: doc_freq for word, doc_freq in zip(df[0], df[3])}
        return {"wikipedia-word-freq": word_freq, "wikipedia-doc-freq": doc_freq}

    @staticmethod
    def subtlex_uk_freq():
        df = pd.read_csv(Path.SUBTLEX_UK_FREQ, usecols=[0, 1, 2, 3, 4]).fillna(0)
        words = df.iloc[:, 0]
        ret = {}
        keys = [
            "subtlex-uk-freq",
            "subtlex-uk-cbeebies",
            "subtlex-uk-cbbc",
            "subtlex-uk-bnc",
        ]
        for key, i in zip(keys, range(1, len(df.columns))):
            ret[key] = {word: freq for word, freq in zip(words, df.iloc[:, i])}
        return ret

    @staticmethod
    def subtlex_us_freq():
        df = pd.read_csv(Path.SUBTLEX_US_FREQ)
        words = df.iloc[:, 0]
        ret = {}
        keys = [
            "subtlex-us-freq",
            "subtlex-us-num-films",
            "subtlex-us-freq-low",
            "subtlex-us-num-films-low",
            "subtlex-us-freq-mil",
            "subtlex-us-freq-log",
            "subtlex-us-percentage-films",
            "subtlex-us-num-films-log",
        ]
        for key, i in zip(keys, range(1, len(df.columns))):
            ret[key] = {word: freq for word, freq in zip(words, df.iloc[:, i])}
        return ret
