import unittest
from util import *

# Add your import statements here
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer


class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """
        pattern = r'\s+'
        regexp = RegexpTokenizer(pattern, gaps=True)

        tokenizedText = []
        for setence in text:
            tokenizedText.append(regexp.tokenize(setence .replace(',', ' , ')))

        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        for sentence in text:
            tokenizedText.append(TreebankWordTokenizer().tokenize(sentence))

        return tokenizedText



################################## Unit Testig #############################################

class TokenizationTestMethods(unittest.TestCase):

    def test_naive(self):
        # Arrange
        text = ["There are multiple ways we can perform tokenization on given text data",
                "We can choose any method based on langauge, library and purpose of modeling"]

        excepted_result = [['There', 'are', 'multiple', 'ways', 'we', 'can', 'perform', 'tokenization', 'on', 'given', 'text', 'data'],
                           ['We', 'can', 'choose', 'any', 'method', 'based', 'on', 'langauge',',', 'library', 'and', 'purpose', 'of', 'modeling']]
        # Act
        actual_result = Tokenization().naive(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

    def test_pennTreeBank(self):
        # Arrange
        text = ["Good muffins cost $3.88\nin New York",
                "Please buy me\ntwo of them.\nThanks"]

        excepted_result = [['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York'],
                           ['Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks']]
        # Act
        actual_result = Tokenization().pennTreeBank(text)
        # Assertion
        self.assertEqual(excepted_result, actual_result)


if __name__ == '__main__':
    unittest.main()

