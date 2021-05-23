from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
# Downloads an initial pkg to avoid tokenize errors
# nltk.download('punkt')

stemmer = PorterStemmer()


# splits sentence into array of words/tokens
def tokenize(sentence):
    # token can be a word, punctuation character, or number
    return nltk.word_tokenize(sentence)


# Removes letters at the end to display root
def stem(word):
    return stemmer.stem(word.lower())

# testing
    # a = "How long does shipping take?"
    # print(a)
    # a = tokenize(a)
    # print(a)

    # words = ["Organize", "organizes", "organizing"]
    # stemmed_words = [stem(w) for w in words]
    # print(stemmed_words)


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

# Testing to ensure training works
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# testing = bag_of_words(sentence, words)
# print(testing)
