import re
import numpy as np

from gensim import corpora
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("./models/wiki-news-300d-1M.vec")

model.similar_by_word("fake")
model.similar_by_word("rip-off")

with open("sample.txt", encoding="utf-8") as f:
    data = f.read()

OOV_WORDS = ["10fold", "fastfact"]


def prepare_document_words():
    doc_words = list(data.lower().split())

    stoplist = set('for a of the and to in'.split())
    doc_words = [word for word in doc_words if word not in stoplist]
    doc_words = [re.sub('[^A-Za-z0-9]+', '', word) for word in doc_words]
    doc_words = [word for word in doc_words if word not in OOV_WORDS]
    doc_words = list(set(doc_words))

    return doc_words


def prepare_sentences():
    texts = data.split("\n")
    texts = [x for x in texts if x != ""]

    sentences = list(map(lambda x: x.lower().split(), texts))

    stoplist = set('for a of the and to in'.split())
    sentences = [[word for word in _word if word not in stoplist] for _word in sentences]
    sentences = [[re.sub('[^A-Za-z0-9]+', '', word) for word in _word] for _word in sentences]

    dictionary = corpora.Dictionary(sentences)
    print(dictionary.token2id)
    dictionary.save("news.dict")

    return sentences


sentence_tokens = prepare_sentences()


def make_vector_array_from_words(words):
    vec_array = np.zeros((len(words), 300))
    for i, w in enumerate(words):
        vec = model.get_vector(w)
        vec_array[i, :] = vec
    print(vec_array.shape)

    return vec_array


make_vector_array_from_words(sentence_tokens[0])


def similar_words_with_scores(query_word, words_list, threshold=0.5):
    idx2word = {}
    word2idx = {}
    for i, item in enumerate(words_list):
        idx2word[i] = item
        word2idx[item] = i

    try:

        similarity_scores = [
            {
                i: model.similarity(query_word, words_list[i])
            }
            for i in idx2word.keys()
        ]

        similar_words_all_scores = [
            {
                idx2word[k]: v
            }
            for score in similarity_scores
            for k, v in score.items()
        ]
        print("All words scores", similar_words_all_scores)

        similar_words_with_threshold = [
            {
                idx2word[k]: v
            }
            for score in similarity_scores
            for k, v in score.items()
            if v > threshold
        ]
        print("Similar words for `%s` with threshold = %s : %s " % (query_word,
                                                                    threshold,
                                                                    similar_words_with_threshold))
    except KeyError:
        print("Word = %s is not in the vocabulary, moving on..")


doc_words = prepare_document_words()
similar_words_with_scores("fake", doc_words, threshold=0.35)
