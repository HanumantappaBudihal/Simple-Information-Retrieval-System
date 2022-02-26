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

import unittest

class StopwordRemovalTestMethods(unittest.TestCase):

    def test_fromList(self):
        # Arrange
        text = ["He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and fishery rihgts at once.",
		        "He was the more ready to do this becuase the rights had become much less valuable, and he had indeed the vaguest idea where the wood and river in question were."]

        excepted_result = ["He determined drop litigation monastry, relinguish claims wood-cuting fishery rihgts.",
						   "He ready becuase rights become much less valuable, indeed vaguest idea wood river question."]
        # Act
        actual_result = StopwordRemoval().fromList(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()
