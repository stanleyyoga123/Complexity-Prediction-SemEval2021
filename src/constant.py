import os


class Path:
    RESOURCES = "resources"
    DATA = "data"
    SCRATCH = "scratch"
    PRETRAINED = "pretrained"
    MODEL = "models"

    TRAIN_SINGLE = os.path.join(DATA, "single_train.csv")
    DEV_SINGLE = os.path.join(DATA, "single_dev.csv")
    TEST_SINGLE = os.path.join(DATA, "single_test.csv")

    TRAIN_MULTI = os.path.join(DATA, "multi_train.csv")
    DEV_MULTI = os.path.join(DATA, "multi_dev.csv")
    TEST_MULTI = os.path.join(DATA, "multi_test.csv")

    CONFIG_SCRATCH = os.path.join(RESOURCES, "scratch_config.yaml")
    CONFIG_PRETRAINED = os.path.join(RESOURCES, "pretrained_config.yaml")

    GOOGLE_FREQ = os.path.join(RESOURCES, "google-freq.txt")
    WIKIPEDIA_FREQ = os.path.join(RESOURCES, "wikipedia-freq.txt")
    SUBTLEX_US_FREQ = os.path.join(RESOURCES, "subtlex-us.csv")
    SUBTLEX_UK_FREQ = os.path.join(RESOURCES, "subtlex-uk.csv")