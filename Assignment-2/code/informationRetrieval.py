from util import *
import numpy as np
from collections import defaultdict
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
        m = len(docIDs)
        for i in range(m):
            docIDs[i] -= 1

        N = docIDs[-1]

        words = []
        for doc in docs:
            for sent in doc:
                for word in sent:
                    word = word.lower()
                    if word.isalpha():
                        words.append(word)
                    else:
                        if '-' in word:
                            words.extend(word.split('-'))

        wordCntDoc = [defaultdict(int) for _ in range(N+1)]               
        words = [wordnet.synsets(word)[0].name() if len(wordnet.synsets(word)) > 0 else word for word in words]
        unq_words = list(set(words))
        
        for idx in docIDs:
            for sent in docs[idx]:
                for word in sent:
                    word = word.lower()
                    if word.isalpha():
                        if len(wordnet.synsets(word)) > 0:
                            wordCntDoc[idx][wordnet.synsets(word)[0].name()] += 1
                        else:
                            wordCntDoc[idx][word] += 1
                    else:
                        if '-' in word:
                            w1, w2 = word.split('-')
                            if len(wordnet.synsets(w1)) > 0:
                                wordCntDoc[idx][wordnet.synsets(w1)[0].name()] += 1
                            else:
                                wordCntDoc[idx][w1] += 1
                            if len(wordnet.synsets(w2)) > 0:
                                wordCntDoc[idx][wordnet.synsets(w2)[0].name()] += 1
                            else:
                                wordCntDoc[idx][w2] += 1

        df = [0] * (len(unq_words)+1)
        gf = [0] * (len(unq_words)+1)
        idf = [0] * (len(unq_words)+1)
        entropy = [1] * (len(unq_words)+1)

        for i, word in enumerate(unq_words):
            df[i] = len([idx for idx in docIDs if wordCntDoc[idx][word] != 0])

        for i, word in enumerate(unq_words):
            for idx in docIDs:
                gf[i] += wordCntDoc[idx][word]      

        for i, n in enumerate(df):
            idf[i] = np.log((N+1)/(n+1))

        for i, word in enumerate(unq_words):
            for idx in docIDs:
                pij = wordCntDoc[idx][word]/gf[i]
                entropy[i] += (np.log(pij+1) * pij)/np.log(N)

        index = np.zeros((len(unq_words), N+1))
        for doc in docIDs:
            for i, word in enumerate(unq_words):
                index[i][doc] = (entropy[i] * (1 + np.log(wordCntDoc[doc][word]))) if wordCntDoc[doc][word] > 0 else 0

        self.k = 300
        U, s, Vh = np.linalg.svd(index)
        self.s = np.zeros((U.shape[0], Vh.shape[0]))
        self.sigma = np.zeros((self.k, self.k))

        for i in range(self.k):
            self.s[i][i] = s[i]
            self.sigma[i][i] = s[i]

        self.U = np.dot(U, self.s)
        self.U = self.U[:, :self.k]
        self.Vh = np.dot(self.s, Vh)
        self.Vh = self.Vh[:self.k]

        self.wordCntDoc = wordCntDoc
        self.docs = docs
        self.docIDs = docIDs
        self.idf = idf
        self.unq_words = unq_words
        self.index = index

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

        # Getting Vector for Query
        doc_IDs_ordered = []
        for query in queries:
            vec = np.zeros((len(self.unq_words), 1))
            cnt = defaultdict(int)

            for sent in query:
                for word in sent:
                    word = word.lower()
                    if word.isalpha():
                        if len(wordnet.synsets(word)) > 0:
                            cnt[wordnet.synsets(word)[0].name()] += 1
                        else:
                            cnt[word] += 1
                    else:
                        if '-' in word:
                            w1, w2 = word.split('-')
                            if len(wordnet.synsets(w1)) > 0:
                                cnt[wordnet.synsets(w1)[0].name()] += 1
                            else:
                                cnt[w1] += 1
                            if len(wordnet.synsets(w2)) > 0:
                                cnt[wordnet.synsets(w2)[0].name()] += 1
                            else:
                                cnt[w2] += 1

            for i, word in enumerate(self.unq_words):
                vec[i][0] = self.idf[i]*(cnt[word]) if cnt[word] > 0 else 0
            
            scores = [[0, 0]] * (self.docIDs[-1]+1)
            vec = np.linalg.multi_dot([np.linalg.inv(self.sigma), self.U.T, vec])

            for idx in self.docIDs:
                sc = np.dot(vec.T, self.Vh[:, idx].reshape(self.k, 1))
                if not np.linalg.norm(vec) or not np.linalg.norm(self.Vh[:, idx]):
                    scores[idx] = [0, idx]
                    continue
                sc = sc / np.linalg.norm(vec)
                sc = sc / np.linalg.norm(self.Vh[:, idx])
                scores[idx] = [sc, idx]
            
            scores.sort(reverse=True)
            doc_IDs_ordered.append([idx+1 for (_, idx) in scores])

        return doc_IDs_ordered
