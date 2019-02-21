"""TODO"""
from argparse import ArgumentParser

from numpy import inf

from model import SimpleModel
from utils.io_utils import IOUtils

def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument("--train",
                        dest="train", help="Path to train file", metavar="FILE", default="N/A")
    parser.add_argument("--train-type",
                        dest="train_file_type", help="Train file type", default="csv")
    parser.add_argument("--prevectors",
                        dest="external_embedding",
                        help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_argument("--prevectype",
                        dest="external_embedding_type",
                        help="Pre-trained vector embeddings type", default=None)
    parser.add_argument("--wembedding",
                        type=int, dest="wembedding_dims", default=300)
    parser.add_argument("--lstmdims",
                        type=int, dest="lstm_dims", default=128)
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

    model = SimpleModel(w2i, permissions, args)

    train_data, test_data = model.train_test_split(args.train)
    similarities = model.train_unsupervised(train_data)
    stats = model.statistics(similarities)
    draw_histogram(stats, "unsupervised.png")

if __name__ == '__main__':
    main()
