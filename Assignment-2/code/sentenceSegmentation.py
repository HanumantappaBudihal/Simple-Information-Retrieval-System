from util import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		Sentences = []
		partial_segmented = text.split('.')
		
		for j in partial_segmented : 
			j_split = j.split('?')
			for k in j_split :
				k_split = k.split('!')
				[Sentences.append(l.strip()) for l in k_split if not l.strip() == ""] 
		
		return Sentences

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		Sentences = None
		Sentences= PunktSentenceTokenizer().tokenize(text)

		return Sentences

################################## Unit Testig #############################################

import unittest

class SentenceSegmentationTestMethods(unittest.TestCase):

    def test_naive(self):
        # Arrange
        text = "Hello everyone. Welcome to IIT Madras. You are studying NLP"
        excepted_result = ['Hello everyone',
                           'Welcome to IIT Madras',
                           'You are studying NLP']
        # Act
        actual_result = SentenceSegmentation().naive(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

    def test_punkt(self):
        # Arrange
        text = """(How does it deal with this parenthesis?)  "It should be part of the previous sentence." 
				"(And the same with this one.)" ('And this one!') "('(And (this)) '?)" [(and this. )]"""

        excepted_result = ['(How does it deal with this parenthesis?)',
                           '"It should be part of the previous sentence."',
                           '"(And the same with this one.)"',
                           "('And this one!')",
                           "\"('(And (this)) '?)\"",
                           '[(and this. )]' ]
        # Act
        actual_result = SentenceSegmentation().punkt(text)
        # Assertion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()
