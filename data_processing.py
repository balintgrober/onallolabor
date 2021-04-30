import string
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import spacy
import gensim
from gensim import corpora, similarities

def avg_review_length(reviews):
    lengths = []
    number_of_reviews = len(reviews)
    for review in reviews:
        length = len(review)
        lengths.append(length)

    avg_length = float(sum(lengths)) / float(number_of_reviews)
    return avg_length


def avg_word_count(reviews):
    counts = []
    number_of_reviews = len(reviews)
    for review in reviews:
        tokens = word_tokenize(review)
        word_count = len(tokens)
        counts.append(word_count)

    avg_count = float(sum(counts)) / float(number_of_reviews)
    return avg_count


def tokenize_reviews(reviews):
    tokenized_reviews = []
    for review in reviews:
        tokenized_review = word_tokenize(review)
        for word in tokenized_review:
            tokenized_reviews.append(word)

    return tokenized_reviews


def lemmatize_words(docs, stop_words):
    lemmatized_reviews = []
    for doc in docs:
        lemmatized_sentence = " ".join([token.lemma_ for token in doc if token.lemma_.lower() not in stop_words and token.lemma_ not in string.punctuation])
        lemmatized_reviews.append(lemmatized_sentence)

    return lemmatized_reviews


def word_count(words):
    word_freq = nltk.FreqDist(words)

    return word_freq


def remove_html_tags(texts):
    texts_without_html = []
    for text in texts:
        text = BeautifulSoup(text, "html.parser").text
        text = re.sub(r"http[s]?://\S+", "", text)
        #text = re.sub(r"\s+", " ", text)
        texts_without_html.append(text)

    return texts_without_html



def similarity(corpus, dictionary):
    index = similarities.MatrixSimilarity(corpus, num_features=len(dictionary))
    return index


def process(imdb_df):
    nlp = spacy.load("en_core_web_sm")

    positive = imdb_df.loc[imdb_df["sentiment"] == "positive", "review"]
    negative = imdb_df.loc[imdb_df["sentiment"] == "negative", "review"]

    positive_reviews = positive.tolist()
    negative_reviews = negative.tolist()

    positive_reviews = remove_html_tags(positive_reviews)
    negative_reviews = remove_html_tags(negative_reviews)

    docs_positive = list(nlp.pipe(positive_reviews, disable=["parser", "ner"]))
    docs_negative = list(nlp.pipe(negative_reviews, disable=["parser", "ner"]))

    average_positive_length = avg_review_length(positive_reviews)
    average_negative_length = avg_review_length(negative_reviews)
    print("Average length of positive reviews:", average_positive_length, "characters")
    print("Average length of negative reviews:", average_negative_length, "characters")

    word_count_positive = avg_word_count(positive_reviews)
    word_count_negative = avg_word_count(negative_reviews)
    print("Average length of positive reviews:", word_count_positive, "words")
    print("Average length of negative reviews:", word_count_negative, "words")

    stop_words = set(stopwords.words("english"))

    lemmatized_positive_reviews = lemmatize_words(docs_positive, stop_words)
    lemmatized_negative_reviews = lemmatize_words(docs_negative, stop_words)

    tokenized_positive_reviews = tokenize_reviews(lemmatized_positive_reviews)
    tokenized_negative_reviews = tokenize_reviews(lemmatized_negative_reviews)

    positive_word_freq = word_count(tokenized_positive_reviews)
    negative_word_freq = word_count(tokenized_negative_reviews)

    print(positive_word_freq.most_common(10))
    print(negative_word_freq.most_common(10))

    lemmatized_positive_docs = list(nlp.pipe(lemmatized_positive_reviews, disable=["parse", "ner", "lemmatizer"]))
    lemmatized_negative_docs = list(nlp.pipe(lemmatized_negative_reviews, disable=["parse", "ner", "lemmatizer"]))

    positive_texts = [[token.text for token in doc] for doc in lemmatized_positive_docs]
    negative_texts = [[token.text for token in doc] for doc in lemmatized_negative_docs]

    first_10_positive = positive_texts[:10]
    first_10_negative = negative_texts[:10]

    positive_dictionary = corpora.Dictionary(first_10_positive)
    negative_dictionary = corpora.Dictionary(first_10_negative)
    positive_corpus_10 = [positive_dictionary.doc2bow(text) for text in first_10_positive]
    negative_corpus_10 = [negative_dictionary.doc2bow(text) for text in first_10_negative]

    index_pos = similarity(positive_corpus_10, positive_dictionary)
    index_neg = similarity(negative_corpus_10, negative_dictionary)

    for sim in index_pos:
        print(sim)

    for sim in index_neg:
        print(sim)