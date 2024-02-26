import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import re
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

PATH = r'../data/sentiment_tweets3.csv'

def load_data():
    df = pd.read_csv(PATH, encoding='UTF-8')
    df = df.drop(columns=['Index'])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    return df

# tokenize text
def tokenize_text(text):
    # Apply word_tokenize function to each element in the 'message to examine' column
    text['message to examine'] = text['message to examine'].apply(lambda x: word_tokenize(x))
    # Return the modified DataFrame with tokenized text
    return text

token_list = tokenize_text(load_data())

# removing characters from sentences
def remove_characters_after_tokenization(tokens):
    # Compile a regular expression pattern to match any punctuation characters
    PATTERN = re.compile('[{}]'.format(re.escape(string.punctuation)))
    # Remove punctuation characters from each token using the compiled pattern
    # Filter out empty tokens after removing punctuation
    filtered = filter(None, [PATTERN.sub('', token) for token in tokens])
    return filtered

    # convert filter object to list
token_list['message to examine'] = [list(filter(None, remove_characters_after_tokenization(tokens)))
                                    # Iterate over tokenized sentences in the 'message to examine' column
                                    for tokens in token_list['message to examine']]


def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    filtered_tokens = [token for token in text if token not in stopwords_list]
    return filtered_tokens

token_list['message to examine'] = [remove_stopwords(text) for text in token_list['message to examine']]


def remove_repeated_characters(token):
    repeat_pattern = re.compile(r'(\W*)(\w)\2(\w*)')
    match_subtitutions = r'\1\2\3'  # creating 3 groups, used for the repeating word

    # define a recursive function for repeated words
    def replace(old_word):
        if wordnet.synsets(old_word):   # check if word has a synset in wordnet
            return old_word
        # Replace repeated characters with instance
        new_word = repeat_pattern.sub(match_subtitutions, old_word)
        # Recursively call replace until no changes are made or if a synset is found
        return replace(new_word) if new_word != old_word else new_word
    # Apply replace functon to every word
    corrected_word = [replace(word) for word in token]
    return corrected_word   # return the list with the corrected words

token_list['message to examine'] = [list(filter(None, remove_repeated_characters(token))) for token in
                                    token_list['message to examine']]

# Stemming involves reducing words to their root or base form
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

token_list['message to examine'] = [[stem_words(token) for token in sentences] for sentences
                                    in token_list['message to examine']]

# Create POS tagging here...

# Create Shallow Parser to extract meaningful chunks out of sentences...
