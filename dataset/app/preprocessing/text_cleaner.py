import nltk
from nltk.corpus import stopwords, wordnet
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


class TextCleaner:
    stemmer: SnowballStemmer('english')
    lemmatizer: WordNetLemmatizer()
    stopwords = stopwords.words('english')

    def __init__(self,
                 name: str,
                 decontrac=False,
                 remove_stop_words=False,
                 stem=False,
                 lemmatize=False
                 ):
        self.name = name
        self.decontract = decontrac
        self.remove_stop_words = remove_stop_words
        self.stem = stem
        self.lemmatize = lemmatize

    @staticmethod
    def _decontract(sentence):
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence

    def _remove_stop_words(self, sentence):
        tokens = sentence.split()
        cleaned_tokens = [i for i in tokens if i not in self.stopwords]
        return ' '.join(cleaned_tokens)

    def _stem(self, sentence: str):
        return

    @staticmethod
    def _get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def _lemmatize(self, sentence):
        return [self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word)) for word in sentence]

    def preprocess_text(self, data: [str]) -> [str]:
        processed_text = []
        for message in data:
            processed_message = message.lower()

            processed_message = processed_message if processed_message.find('<a href') == -1 else \
            processed_message.split('<a href')[0]
            processed_message = re.sub(r'https?://[a-z\d./]+', ' ', processed_message)
            processed_message = re.sub(r'', '', processed_message)
            processed_message = re.sub(r'[^a-zA-Z\dàáâçéèêëîïôûùüÿñæœ]', ' ', processed_message)
            processed_message = re.sub(r'[\s{2}]+', ' ', processed_message)

            # language agnostic BERT has cased vocabulary
            if self.decontract:
                processed_message = self._decontract(processed_message)
            if self.remove_stop_words:
                processed_message = self._remove_stop_words(processed_message)
            if self.stem:
                processed_message = [self.stemmer.stem(i) for i in processed_message]
            if self.lemmatize:
                processed_message = self._lemmatize(processed_message)

            processed_text.append(processed_message)
        return processed_text


basic_text_cleaner = TextCleaner(
    name='basic_text_cleaner',
)

decontracting_text_cleaner = TextCleaner(
    name='decontracting_text_cleaner',
    decontrac=True
)

stop_words_removing_text_cleaner = TextCleaner(
    name='stop_words-removing_text_cleaner',
    decontrac=True,
    remove_stop_words=True
)

stemming_text_cleaner = TextCleaner(
    name='stemming_text_cleaner',
    decontrac=True,
    remove_stop_words=True,
    stem=True,
)

lammatizing_text_cleaner = TextCleaner(
    name='lemmatizingtext_cleaner',
    decontrac=True,
    remove_stop_words=True,
    stem=True,
    lemmatize=True
)

"""
In practice we want to use only basic_text_cleaner.
Since a lot of preprocessing is already handled by BERT pre-processing layer
But let's just leave them, and compare resulting accuracy later on
"""
