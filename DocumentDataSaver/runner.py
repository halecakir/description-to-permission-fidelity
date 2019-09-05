"""TODO"""
from argparse import ArgumentParser
from document_dump import run


def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument(
        "--saved-data",
        dest="saved_data",
        help="Saved embeddings, train/test & vocab data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--stemmer", dest="stemmer", help="Apply word stemmer", default="porter"
    )
    args = parser.parse_args()
    return args


def main():
    """TODO"""
    args = parse_arguments()
    run(args)


if __name__ == "__main__":
    main()
