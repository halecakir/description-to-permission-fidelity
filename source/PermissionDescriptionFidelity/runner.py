from optparse import OptionParser
from utils import Utils
from model import SimpleModel

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="Path to apps csv train file", metavar="FILE", default="N/A")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    (options, args) = parser.parse_args()

    print ('Extracting vocabulary')
    words, w2i, p2i = Utils.vocab(options.train)


    model = SimpleModel(words, w2i, p2i, options)

    model.train(options.train)