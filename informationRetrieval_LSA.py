import numpy as np
from nltk.corpus import wordnet

class InformationRetrieval():

    def __init__(self):
        self.index = None

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
        inv = dict((v, k) for (k, v) in enumerate(unique_words))
        index = np.zeros((len(unique_words), N))

        for idx in docIDs:
            for sent in docs[idx]:
                for word in sent:
                    word = word.lower()
                    if '-' in word:
                        for w in word.split('-'):
                            index[inv[w]][idx] += 1
                    else:
                        index[inv[word]][idx] += 1

        idf = np.zeros((len(unique_words), 1))
        for i, word in enumerate(unique_words):
            n = index[inv[word]].sum()
            idf[i][0] = np.log((N + 1)/(n + 1))
        
        # create tf-idf matrix and then apply svd to it
        index *= idf

        self.k = 550
        U, s, V = np.linalg.svd(index)
        self.s = np.zeros((U.shape[0], V.shape[0]))
        self.sigma = np.zeros((self.k, self.k))

        for i in range(self.k):
            self.s[i][i] = s[i]
            self.sigma[i][i] = s[i]

        self.U = np.dot(U, self.s)[:, :self.k]
        self.V = np.dot(self.s, V)[:self.k]

        self.docIDs = docIDs
        self.idf = idf
        self.unique_words = unique_words
        self.index = index
        self.inv = inv


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
            w = np.zeros((len(self.unique_words), 1))

            for sent in query:
                for word in sent:
                    if '-' in word:
                        for p in word.split('-'):
                            try:
                                w[self.inv[p]][0] += 1
                            except KeyError:
                                pass
                    else:
                        try:
                            w[self.inv[word]][0] += 1
                        except KeyError:
                            pass

            w *= self.idf
            w_proj = np.linalg.multi_dot([np.linalg.inv(self.sigma), self.U.T, w])

            for idx in self.docIDs:
                dot_prod = np.dot(w_proj.T, self.V[:, idx].reshape(self.k, 1))
                norms_prod = np.linalg.norm(w_proj) * np.linalg.norm(self.V[:, idx])
                retrieved_docs[self.docIDs[idx]+1] = dot_prod/(norms_prod + 1e-8)
            
            doc_IDs_ordered.append(sorted(retrieved_docs, reverse=True, key=lambda x: retrieved_docs[x]))

        return doc_IDs_ordered
