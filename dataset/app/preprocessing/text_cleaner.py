import re


class TextCleaner:

    def __init__(self, name: str):
        self.name = name

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

    @staticmethod
    def remove_html_elements(x: str) -> str:
        if x.find('<a href') == -1:
            return x
        else:
            return x.split('<a href')[0]

    def preprocess_text(self, data: [str]) -> [str]:
        processed_text = []
        for message in data:
            processed_message = message.lower()
            processed_message = self.remove_html_elements(processed_message)
            processed_message = re.sub(r'https?://[a-z\d./]+', ' ', processed_message)
            processed_message = re.sub(r'[\s{2}]+', ' ', processed_message)
            processed_text.append(processed_message)
        return processed_text


basic_text_cleaner = TextCleaner(
    name='basic_text_cleaner',
)
