import gensim, logging
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def read_data():
    newsgroups_train = fetch_20newsgroups(shuffle=True, subset='train', remove=('headers', 'footers', 'quotes'))
    text_data = newsgroups_train.data
    target_data = newsgroups_train.target

    return text_data, target_data


texts, labels = read_data()

DICT_FILE = "news.classify.dict"

# slit the text into words
words = list(map(lambda x: x.lower().split(), texts))

# create dict file
dictionary = gensim.corpora.Dictionary(words)
dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=2000000)
dictionary.save(DICT_FILE)
print("dictionary tokenid", len(dictionary.token2id))
print("words", words[:1])

corpus = list(map(lambda x: dictionary.doc2bow(x), words))

print("corp", len(corpus), corpus[:1])

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
print(len(corpus_tfidf))
gensim.corpora.SvmLightCorpus.serialize('svm_20_news_groups.train', corpus_tfidf, labels=labels)
print(corpus_tfidf)
