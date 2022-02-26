
from util import *
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

class InflectionReduction:

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
        """

        # Fill in code here
        lemmatizer = nltk.stem.WordNetLemmatizer()
        reducedText = [[lemmatizer.lemmatize(t)
                        for t in tokens] for tokens in text]

        return reducedText


################################## Unit Testig #############################################

import unittest

class InflectionReductionTestMethods(unittest.TestCase):

    def test_reduce(self):
        # Arrange
        text = ["You are studying NLP",
                "We are building a very good site and I love visiting your site."]

        excepted_result = [['You', 'are', 'study' 'NLP']
                           ['We', 'are', 'build', 'a', 'very', 'good', 'site',
                            'and', 'I', 'love', 'visit', 'your', 'site']]
        # Act
        actual_result = InflectionReduction().reduce(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()
