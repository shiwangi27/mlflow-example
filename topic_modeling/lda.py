import gensim
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import mlflow


def read_data(filename):
    data = pd.read_csv(filename)
    texts = data['texts']
    texts = texts.drop_duplicates()
    print(type(texts))
    return texts.values


news_data = fetch_20newsgroups(shuffle=True, subset="train", remove=("headers", "footers", "quotes"))
texts = news_data.data


def create_dict_and_corpus():
    # split the text into words
    words = list(map(lambda x: x.lower().split(), texts))

    # create dict file
    DICT_FILE = "news.topics.dict"
    dictionary = gensim.corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=2000000)
    dictionary.filter_n_most_frequent(15)
    dictionary.save(DICT_FILE)
    mlflow.log_artifact(DICT_FILE)

    print("dictionary tokenid", len(dictionary.token2id))
    print("words", words[:1])

    corpus = list(map(lambda x: dictionary.doc2bow(x), words))
    print("corpus length", len(corpus))

    return dictionary, corpus


def train(num_topics=5):
    dictionary, corpus = create_dict_and_corpus()

    NUM_TOPICS = num_topics
    ldamodel_file = "news.topics.model"
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                               num_topics = NUM_TOPICS,
                                               id2word=dictionary,
                                               passes=15)
    ldamodel.save(ldamodel_file)
    mlflow.log_artifacts(ldamodel_file)

    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)


# def score():
#     ldamodel = gensim.models.ldamodel.LdaModel.load("news.topics.model")
#     new_doc = "AT&T, Samsung team up to create 5G Innovation Zone in Austin"
#     doc = new_doc.lower().split()
#     topics = ldamodel.get_document_topics(
#         dictionary.doc2bow(doc),
#         minimum_probability=0.3
#     )
#     print(topics)


train()
