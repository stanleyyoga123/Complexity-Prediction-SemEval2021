import os


class Path:
    # RESOURCES = "/raid/data/m13518122/resources"
    # DATA = "/raid/data/m13518122/data"
    # SCRATCH = "scratch"
    # PRETRAINED = "pretrained"
    # MODEL = "/raid/data/m13518122/models"

    RESOURCES = "resources"
    DATA = "data"
    SCRATCH = "scratch"
    PRETRAINED = "pretrained"
    MODEL = "models"

    TRAIN_SINGLE = os.path.join(DATA, "scaled-enhanced-single-train.csv")
    DEV_SINGLE = os.path.join(DATA, "scaled-enhanced-single-dev.csv")
    TEST_SINGLE = os.path.join(DATA, "scaled-enhanced-single-test.csv")

    TRAIN_MULTI = os.path.join(DATA, "scaled-enhanced-multi-train.csv")
    DEV_MULTI = os.path.join(DATA, "scaled-enhanced-multi-dev.csv")
    TEST_MULTI = os.path.join(DATA, "scaled-enhanced-multi-test.csv")

    CONFIG_SCRATCH = os.path.join(RESOURCES, "scratch_config.yaml")
    CONFIG_PRETRAINED = os.path.join(RESOURCES, "pretrained_config.yaml")

    GOOGLE_FREQ = os.path.join(RESOURCES, "google-freq.txt")
    WIKIPEDIA_FREQ = os.path.join(RESOURCES, "wikipedia-freq.txt")
    SUBTLEX_US_FREQ = os.path.join(RESOURCES, "subtlex-us.csv")
    SUBTLEX_UK_FREQ = os.path.join(RESOURCES, "subtlex-uk.csv")
    CONCRETENESS = os.path.join(RESOURCES, "concreteness.txt")
    AGE_OF_ACQUISITION = os.path.join(RESOURCES, "aoa.xlsx")
    COMPLEXITY_SCORE = os.path.join(RESOURCES, "lexicon.tsv")
