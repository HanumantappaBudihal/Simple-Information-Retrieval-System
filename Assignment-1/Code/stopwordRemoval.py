import unittest
from util import *
import nltk
nltk.download("stopwords")


class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        # Fill in code here
        allStopwords = nltk.corpus.stopwords.words("english")
        stopwordRemovedText = [
            [t for t in tokens if not t in allStopwords] for tokens in text]

        return stopwordRemovedText

################################## Unit Testig #############################################


class StopwordRemovalTestMethods(unittest.TestCase):

    def test_fromList(self):
        # Arrange
        text = [['He', 'determined', 'to', 'drop', 'his', 'litigation', 'with', 'the', 'monastry', 'and', 'relinguish', 'his', 'claims', 'to', 'the', 'wood', 'and', 'fishery', 'rihgts', 'at', 'once.'],
                ['He', 'was', 'the', 'more', 'ready', 'to', 'do', 'this', 'becuase', 'the', 'rights', 'had', 'become', 'much', 'less', 'valuable', 'and', 'he', 'had', 'indeed', 'the', 'vaguest', 'idea', 'where', 'the', 'wood', 'and', 'river', 'in', 'question', 'were.']]

        excepted_result = [['He', 'determined', 'drop', 'litigation', 'monastry', 'relinguish', 'claims', 'wood', 'fishery', 'rihgts', 'once.'],
                           ['He', 'ready', 'becuase',  'rights',  'become', 'much', 'less', 'valuable',  'indeed', 'vaguest', 'idea',  'wood',  'river', 'question','were.']]

        # Act
        actual_result = StopwordRemoval().fromList(text)
        # Assertion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()
