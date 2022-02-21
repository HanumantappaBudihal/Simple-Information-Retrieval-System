from util import *
import nltk
nltk.download("stopwords")

# Add your import statements here
# Varun Gumma


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

		#Fill in code here
		allStopwords = nltk.corpus.stopwords.words("english")
		stopwordRemovedText = [[t for t in tokens if not t in allStopwords] for tokens in text]

		return stopwordRemovedText




	