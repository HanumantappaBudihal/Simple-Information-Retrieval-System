
import unittest
from nltk.stem import SnowballStemmer
from util import *
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")


class InflectionReduction:

    def is_number(self, s):
        try:
            float(s)
            return 'num11'
        except ValueError:
            try:
                float(s.replace(',', '.'))
                return 'num11'
            except ValueError:
                return s

    def reduce(self, text):
        """
                Stemming/Lemmatization

                Parameters
                ----------
                arg1 : list
                        A list of lists where each sub-list a sequence of tokens
                        representing a sentence

                Returns
                -------
                list
                        A list of lists where each sub-list is a sequence of
                        stemmed/lemmatized tokens representing a sentence

                we are using snowball stemmer
                """

        sb = SnowballStemmer('english')
        reducedText = [[self.is_number(sb.stem(word).replace('/', '').replace('-', '')) for word in sentence]
                       for sentence in text]

        # Fill in code here

        return reducedText
################################## Unit Testig #############################################


class InflectionReductionTestMethods(unittest.TestCase):

    def test_reduce(self):
        # Arrange
        text = [['You', 'are', 'studying', 'NLP'],
                ['We', 'are', 'building', 'a', 'very', 'good', 'site', 'and', 'I', 'love', 'visiting', 'your', 'site.']]

        excepted_result = [['you', 'are', 'studi', 'nlp'],
                           ['we', 'are', 'build', 'a', 'veri', 'good', 'site', 'and', 'i', 'love', 'visit', 'your', 'site.']]

        # Act
        actual_result = InflectionReduction().reduce(text)
        # Assertion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()
