"""TODO"""
from argparse import ArgumentParser

from numpy import inf

from model.rnn_model import RNNModel
from utils.io_utils import IOUtils

def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument("--train",
                        dest="train",
                        help="Path to train file",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--train-type",
                        dest="train_file_type",
                        help="Train file type",
                        default="csv")
    parser.add_argument("--prevectors",
                        dest="external_embedding",
                        help="Pre-trained vector embeddings",
                        metavar="FILE")
    parser.add_argument("--prevectype",
                        dest="external_embedding_type",
                        help="Pre-trained vector embeddings type",
                        default=None)
    parser.add_argument("--wembedding",
                        type=int,
                        dest="wembedding_dims",
                        default=300)
    parser.add_argument("--lstmdims",
                        type=int,
                        dest="lstm_dims",
                        default=128)
    parser.add_argument("--sequence-type",
                        dest="sequence_type",
                        help="Train sequence type e.g. raw, windowed, dependency, chunk",
                        default="windowed")
    parser.add_argument("--window-size",
                        dest="window_size",
                        type=int,
                        help="Window size for windowed sequence",
                        default=2)
    args = parser.parse_args()
    return args

def draw_histogram(stats, img_name):
    """TODO"""
    related_all = []
    unrelated_all = []
    for doc_id in stats:
        related_all.extend([i for i in stats[doc_id]["related"]["all"] if i > -inf])
        unrelated_all.extend([i for i in stats[doc_id]["unrelated"]["all"] if i > -inf])

    from matplotlib import pyplot

    pyplot.title("All similarity")
    pyplot.hist(related_all, bins='auto', alpha=0.5, label='related')
    pyplot.hist(unrelated_all, bins='auto', alpha=0.5, label='unrelated')
    pyplot.legend(loc='upper right')
    pyplot.savefig(img_name)
    pyplot.clf()

def main():
    """TODO"""
    args = parse_arguments()
    print('Extracting vocabulary')
    _, w2i, permissions = IOUtils.vocab(args.train, file_type=args.train_file_type, lower=True)

    model = RNNModel(w2i, permissions, args)

    train_data, test_data = IOUtils.train_test_split(args.train,
                                                     args.train_file_type,
                                                     args.sequence_type,
                                                     args.window_size)
    similarities = model.train_unsupervised(train_data)
    model.train_supervised(train_data)
    model.test(train_data)
    stats = model.statistics(similarities)
    draw_histogram(stats, "unsupervised.png")

if __name__ == '__main__':
    main()
