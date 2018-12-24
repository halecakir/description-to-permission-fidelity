from optparse import OptionParser
from utils import Utils
from model import SimpleModel

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

    similarities = model.train(options.train)
    stats = model.statistics(similarities)
    
    related_max = []
    unrelated_max = []

    related_avg = []
    unrelated_avg = []

    related_all = []
    unrelated_all = []
    for doc_id in stats:
        related_max.append(stats[doc_id]["related"]["max"])
        related_avg.append(stats[doc_id]["related"]["avg"])
        unrelated_max.append(stats[doc_id]["unrelated"]["max"])
        unrelated_avg.append(stats[doc_id]["unrelated"]["avg"])
        related_all.extend(stats[doc_id]["related"]["all"])
        unrelated_all.extend(stats[doc_id]["unrelated"]["all"])

    from matplotlib import pyplot
    pyplot.title("Max similarity") 
    pyplot.hist(related_max, bins='auto', alpha=0.5, label='related')
    pyplot.hist(unrelated_max, bins='auto', alpha=0.5, label='unrelated')
    pyplot.legend(loc='upper right')
    pyplot.savefig("max_sim.png")

    pyplot.clf()

    pyplot.title("Avg similarity") 
    pyplot.hist(related_avg, bins='auto', alpha=0.5, label='related')
    pyplot.hist(unrelated_avg, bins='auto', alpha=0.5, label='unrelated')
    pyplot.legend(loc='upper right')
    pyplot.savefig("avg_sim.png")

    pyplot.clf()

    pyplot.title("All similarity") 
    pyplot.hist(related_all, bins='auto', alpha=0.5, label='related')
    pyplot.hist(unrelated_all, bins='auto', alpha=0.5, label='unrelated')
    pyplot.legend(loc='upper right')
    pyplot.savefig("all_sim.png")
