import numpy as np
from numpy.linalg import norm

class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndexWithSVD(self, docs, docIDs, n_comp=750):
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

        # compute idf with smoothening
        idf = np.zeros(len(unique_words))
        for i, word in enumerate(unique_words):
            n = index[postings[word]].sum()
            idf[i] = np.log((N + 1)/(n + 1)) + 1

        idf = idf.reshape(-1, 1)
        index *= idf

        self.k = n_comp
        U, s, Vt = np.linalg.svd(index)
        index_recon = np.linalg.multi_dot([U[:, :self.k], np.diag(s[:self.k]), Vt[:self.k]])

        self.docIDs = docIDs
        self.idf = idf
        self.unique_words = unique_words
        self.index = index_recon
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
            q_vec = np.zeros(len(self.unique_words))

            for sent in query:
                for word in sent:
                    if '-' in word:
                        for w in word.split('-'):
                            if w in self.postings:
                                q_vec[self.postings[w]] += 1
                    else:
                        if word in self.postings:
                            q_vec[self.postings[word]] += 1

            q_vec = q_vec.reshape(-1, 1)
            q_vec *= self.idf

            dot_prod = np.dot(q_vec.T, self.index)
            norms_prod = (norm(q_vec) * np.array([norm(v) for v in self.index.T])) + 1e-8
            cos_sim = dot_prod / norms_prod

            retrieved_docs = np.squeeze(np.argsort(cos_sim, kind="mergesort"), axis=0) + 1
            doc_IDs_ordered.append(retrieved_docs.tolist()[::-1])

        return doc_IDs_ordered
