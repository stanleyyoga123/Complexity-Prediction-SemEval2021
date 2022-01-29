import enchant
import re
import numpy as np
from nltk.corpus import cmudict


class Algorithm:

    def __init__(self):
        self.morpheme_dict = enchant.Dict("en_US")
        self.dictionary = cmudict.dict()
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
        }

    def __manual_syllables(self, word):
        vowel_runs = len(self.VOWEL_RUNS.findall(word))
        exceptions = len(self.EXCEPTIONS.findall(word))
        additional = len(self.ADDITIONAL.findall(word))
        return max(1, vowel_runs - exceptions + additional)

    def count_morphemes(self, word):
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

    def count_syllables(self, word):
        try:
            return max(
                [
                    len(list(y for y in x if y[-1].isdigit()))
                    for x in self.dictionary[word.lower()]
                ]
            )
        except KeyError:
            return self.__manual_syllables(word)

    def count_length(self, word):
        return len(word)

    # def is_acronym(self, word):
        