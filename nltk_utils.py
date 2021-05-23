from nltk.stem.porter import PorterStemmer
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


def bag_of_words(tokenize):
    pass


# testing
# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
