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

		Sentences = None
		Sentences= sent_tokenize(text)

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