from optparse import OptionParser

from numpy import inf

from decorators import logging
from model import SimpleModel
from utils import Utils

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="Path to apps  train file", metavar="FILE", default="N/A")
    parser.add_option("--train-type", dest="train_file_type", help="Train file type", default="csv")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--prevectype", dest="external_embedding_type", help="Pre-trained vector embeddings type", default=None)
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=300)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    (options, args) = parser.parse_args()

    print ('Extracting vocabulary')
    words, w2i, permissions = Utils.vocab(options.train, file_type=options.train_file_type)
    
    model = SimpleModel(words, w2i, permissions, options)
    
    @logging
    def draw_histogram(data, img_name):
        stats = model.statistics(data)   
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

    train_data, test_data = model.train_test_split(options.train)
    similarities = model.train_unsupervised(test_data)
    draw_histogram(similarities, "unsupervised.png".format(0))

    """    
    train_data, test_data = model.train_test_split(options.train)
    similarities = model.test(test_data)
    draw_histogram(similarities, "trained_epoch_{}.png".format(0))
    model.test(test_data)

    for i in range(10):
        print("Epoch {}".format(i+1))
        model.train(train_data)
        similarities = model.test(test_data)
        draw_histogram(similarities, "trained_epoch_{}.png".format(i+1))
    """