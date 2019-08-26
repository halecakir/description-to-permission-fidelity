"""TODO"""
from argparse import ArgumentParser

from numpy import inf
from scripts.similarity_experiment import SimilarityExperiment
import scripts.roc_experiment_document_based as roc_experiment

from model.rnn_model import RNNModel
from utils.io_utils import IOUtils


def parse_arguments():
    """TODO"""
    parser = ArgumentParser()
    parser.add_argument("--permission-type",
                        dest="permission_type",
                        help="Test permission",
                        default="N/A")
    parser.add_argument("--train",
                        dest="train",
                        help="Path to train file",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--train-type",
                        dest="train_file_type",
                        help="Train file type",
                        default="csv")
    parser.add_argument("--test",
                        dest="test",
                        help="Path to test file",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--test-type",
                        dest="test_file_type",
                        help="Test file type",
                        default="csv")
    parser.add_argument("--prevectors",
                        dest="external_embedding",
                        help="Pre-trained vector embeddings",
                        metavar="FILE")
    parser.add_argument("--prevectype",
                        dest="external_embedding_type",
                        help="Pre-trained vector embeddings type",
                        default=None)
    parser.add_argument("--lstm-type",
                        dest="lstm_type",
                        help="lstm or bilstm",
                        default="lstm")
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
    parser.add_argument("--saved-parameters-dir",
                        dest="saved_parameters_dir",
                        help="Saved model parameters directory",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--saved-prevectors",
                        dest="saved_prevectors",
                        help="Saved model embeddings",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--saved-vocab-test",
                        dest="saved_vocab_test",
                        help="Saved test vocabulary",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--saved-vocab-train",
                        dest="saved_vocab_train",
                        help="Saved train vocabulary",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--saved-sentences-whyper",
                        dest="saved_preprocessed_whyper",
                        help="Saved whyper sentences",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--saved-sentences-acnet",
                        dest="saved_preprocessed_acnet",
                        help="Saved acnet sentences",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--outdir",
                        dest="outdir",
                        help="Output directory",
                        metavar="FILE",
                        default="N/A")
    parser.add_argument("--stemmer",
                        dest="stemmer",
                        help="Apply word stemmer",
                        default="no_stemmer")
    parser.add_argument("--external-info",
                        dest="external_info",
                        help="Add External Info",
                        default="no_info")
    parser.add_argument("--external-info-dim",
                        dest="external_info_dim",
                        type=int,
                        help="External Info Dimension",
                        default=300)

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
    w2i, permissions = IOUtils.load_vocab(args.test,
                                args.test_file_type,
                                args.saved_parameters_dir,
                                args.saved_vocab_test,
                                args.external_embedding,
                                args.external_embedding_type,
                                args.stemmer,
                                True)
    print('RNN Model')

    model = RNNModel(w2i, permissions, args)

    train_data, _ = IOUtils.train_test_split(args.train,
                                             args.train_file_type,
                                             args.sequence_type,
                                             args.window_size)
    similarities = model.train_unsupervised(train_data)
    model.train_supervised(train_data)
    model.test(train_data)
    stats = model.statistics(similarities)
    draw_histogram(stats, "unsupervised.png")

def call_similarity_experiment():
    """TODO"""
    args = parse_arguments()
    print('Extracting vocabulary')
    train_w2i, _ = IOUtils.load_vocab(  args.train,
                                        args.train_file_type,
                                        args.saved_parameters_dir,
                                        args.saved_vocab_train,
                                        args.external_embedding,
                                        args.external_embedding_type,
                                        args.stemmer,
                                        True)

    test_w2i, _ = IOUtils.load_vocab(args.test,
                                     args.test_file_type,
                                     args.saved_parameters_dir,
                                     args.saved_vocab_test,
                                     args.external_embedding,
                                     args.external_embedding_type,
                                     args.stemmer,
                                     True)

    #combine test&train vocabulary
    w2i = train_w2i
    for token in test_w2i:
        if token not in w2i:
            w2i[token] = len(w2i)

    print('Similarity Experiment')
    model = SimilarityExperiment(w2i, args)

    model.run()

def call_roc_pr_auc_experiment():
    args = parse_arguments()

    roc_experiment.run(args)

if __name__ == '__main__':
    call_roc_pr_auc_experiment()
