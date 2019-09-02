"""TODO"""
from argparse import ArgumentParser
from models.base_model import run


def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument(
        "--permission-type",
        dest="permission_type",
        help="Test permission",
        default="N/A",
    )
    parser.add_argument(
        "--saved-parameters-dir",
        dest="saved_parameters_dir",
        help="Saved model parameters directory",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-data",
        dest="saved_data",
        help="Saved embeddings, train/test & vocab data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--saved-review",
        dest="saved_review",
        help="Saved review data",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--model-checkpoint",
        dest="model_checkpoint",
        help="Saved model file",
        metavar="FILE",
        default="N/A",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        help="Output directory",
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
