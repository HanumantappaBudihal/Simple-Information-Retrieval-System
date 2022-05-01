import numpy as np
from numpy.linalg import norm

class InformationRetrieval():

    def __init__(self):
        self.idf = []
        self.postings = None
        self.docIDs = None
        self.unique_words = None
        self.matrix = []
        
    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable
        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        N = len(docIDs)
        for i in range(N):
            docIDs[i] -= 1
        
        words = []
        for doc in docs:
            for sent in doc:
                for word in sent:
                    word = word.lower()
                    if '-' in word:
                        words.extend(word.split('-'))
                    else:
                        words.append(word)

        unique_words = list(set(words))
        postings = dict((v, k) for (k, v) in enumerate(unique_words))

        # compute tf for each word wrt to each document
        index = np.zeros((len(unique_words), N))

        for idx in docIDs:
            for sent in docs[idx]:
                for word in sent:
                    word = word.lower()
                    if '-' in word:
                        for w in word.split('-'):
                            index[postings[w]][idx] += 1
                    else:
                        index[postings[word]][idx] += 1

        # compute idf of each word
        idf = np.zeros(len(unique_words))
        for i, word in enumerate(unique_words):
            n = index[postings[word]].sum()
            idf[i] = np.log((N + 1)/(n + 1))

        idf = idf.reshape(-1, 1)
        # w = tf * idf
        # set all required class variables
        self.matrix = index * idf
        self.unique_words = unique_words
        self.idf = idf
        self.docIDs = docIDs
        self.postings = postings
        
    def rank(self, queries):
        """
        Rank the documents according to relevance for each query
        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        doc_IDs_ordered = []
        for query in queries:
            retrieved_docs = {}
            q_vec = np.zeros(len(self.unique_words))

            for sent in query:
                for word in sent:
                    if '-' in word:
                        for w in word.split('-'):
                            try:
                                q_vec[self.postings[w]] += 1
                            except KeyError:
                                pass
                    else:
                        try:
                            q_vec[self.postings[word]] += 1
                        except KeyError:
                            pass

            q_vec = q_vec.reshape(-1, 1)
            q_vec *= self.idf

            dot_prod = np.dot(q_vec.T, self.matrix)
            norms_prod = (norm(q_vec) * np.array([norm(v) for v in self.matrix.T])) + 1e-8
            cos_sim = dot_prod / norms_prod

            retrieved_docs = np.squeeze(np.argsort(cos_sim, kind="mergesort"), axis=0) + 1
            doc_IDs_ordered.append(retrieved_docs.tolist()[::-1])
            
        return doc_IDs_ordered