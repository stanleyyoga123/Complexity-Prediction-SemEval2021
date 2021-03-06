import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", required=True, help="[fasttext | word2vec | pretrained]"
    )
    parser.add_argument("--prefix", required=True, help="prefix for models")
    parser.add_argument("--config", required=True, help="config for models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.type == "pretrained":
        from src.trainer.pretrained import main

        main(args.config, args.prefix)
    else:
        from src.trainer.scratch import main

        main(args.config, args.type, args.prefix)
