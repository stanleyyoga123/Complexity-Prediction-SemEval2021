from collections import Counter
import enchant
import re
import numpy as np
import textstat
from wordfreq import zipf_frequency
from nltk.corpus import cmudict, stopwords, wordnet, brown
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer


class Algorithm:
    def __init__(self):
        self.morpheme_dict = enchant.Dict("en_US")
        self.cmu_dictionaryy = cmudict.dict()
        self.stopwords_dictionary = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()

        self.brown_dictionary = dict(Counter(brown.words()))

        self.VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
        self.EXCEPTIONS = re.compile("[^aeiou]e[sd]?$|" + "[^e]ely$", flags=re.I)
        self.ADDITIONAL = re.compile(
            "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"
            + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
            flags=re.I,
        )

        self.STR_FUNC_MAP = {
            "count-syllabels": self.count_syllables,
            "count-morphemes": self.count_morphemes,
            "count-length": self.count_length,
            "zipf": self.count_zipf,
            "count-stopword": self.count_stopword,
            "count-synonym": self.count_synonym,
            "count-hypernyms": self.count_hypernyms,
            "count-hyponyms": self.count_hyponyms,
            "is-acronym": self.is_acronym,
            "count-max-dist-hypernyms": self.count_max_dist_hypernyms,
            "count-min-dist-hypernyms": self.count_min_dist_hypernyms,
            "count-dale-chall-index": self.count_dale_chall_index,
            "count-rix-score": self.count_rix_score,
            "count-smog-index": self.count_smog_index,
            "count-lix-score": self.count_lix_score,
            "count-gunning-fox-score": self.count_gunning_fox_score,
            "count-flesch-score": self.count_flesch_score,
            "count-coleman-liau-index": self.count_coleman_liau_index,
            "count-ari-score": self.count_ari_score,
            "count-kincaid-grade-score": self.count_kincaid_grade_score,
            "count-lemma-length": self.count_lemma_length,
            "count-lemma-brown": self.count_lemma_brown,
            "count-freq-brown": self.count_freq_brown,
        }

    def __manual_syllables(self, sentence, word):
        vowel_runs = len(self.VOWEL_RUNS.findall(word))
        exceptions = len(self.EXCEPTIONS.findall(word))
        additional = len(self.ADDITIONAL.findall(word))
        return max(1, vowel_runs - exceptions + additional)

    def count_morphemes(self, sentence, word):
        ctr = 0

        def tokenize(word):
            nonlocal ctr

            if not word:
                return
            for i in range(len(word), -1, -1):
                if self.morpheme_dict.check(word[0:i]):
                    ctr += 1
                    st = word[i:]
                    tokenize(st)
                    break

        tokenize(word)

        return ctr if ctr != 0 else 1

    def count_zipf(self, sentence, word):
        return zipf_frequency(word, "en")

    def count_syllables(self, sentence, word):
        try:
            return max(
                [
                    len(list(y for y in x if y[-1].isdigit()))
                    for x in self.cmu_dictionaryy[word.lower()]
                ]
            )
        except KeyError:
            return self.__manual_syllables(word)

    def count_length(self, sentence, word):
        return len(word)

    def count_hyponyms(self, sentence, word):
        sense = lesk(sentence, word)
        return len(sense.hyponyms())

    def count_hypernyms(self, sentence, word):
        sense = lesk(sentence, word)
        return len(sense.hypernyms())

    def count_max_dist_hypernyms(self, sentence, word):
        sense = lesk(sentence, word)
        return max([len(ss) for ss in sense.hypernym_paths()])

    def count_min_dist_hypernyms(self, sentence, word):
        sense = lesk(sentence, word)
        return min([len(ss) for ss in sense.hypernym_paths()])

    def calculate_wordsense(self, sentence, word):
        return 1

    def count_synonym(self, sentence, word):
        return len(wordnet.synsets(word))

    def count_stopword(self, sentence, word):
        return word in self.stopwords_dictionary

    def is_acronym(self, sentence, word):
        return word.isupper()

    def count_dale_chall_index(self, sentence, word):
        return textstat.dale_chall_readability_score(sentence)

    def count_rix_score(self, sentence, word):
        return textstat.rix(sentence)

    def count_smog_index(self, sentence, word):
        return textstat.smog_index(sentence)

    def count_lix_score(self, sentence, word):
        return textstat.rix(sentence)

    def count_gunning_fox_score(self, sentence, word):
        return textstat.gunning_fog(sentence)

    def count_flesch_score(self, sentence, word):
        return textstat.flesch_reading_ease(sentence)

    def count_coleman_liau_index(self, sentence, word):
        return textstat.coleman_liau_index(sentence)

    def count_ari_score(self, sentence, word):
        return textstat.automated_readability_index(sentence)

    def count_kincaid_grade_score(self, sentence, word):
        return textstat.flesch_kincaid_grade(sentence)

    def count_lemma_length(self, sentence, word):
        return len(self.lemmatizer.lemmatize(word))

    def count_lemma_brown(self, sentence, word):
        lemma = self.lemmatizer.lemmatize(word)
        return self.brown_dictionary.get(lemma, 0)

    def count_freq_brown(self, sentence, word):
        return self.brown_dictionary.get(word, 0)
