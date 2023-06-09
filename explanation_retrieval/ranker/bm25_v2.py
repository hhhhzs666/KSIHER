# From https://github.com/arosh/BM25Transformer/blob/master/bm25.py

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import numpy as np
import scipy.sparse as sp
import spacy
from sklearn import feature_extraction, metrics, pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import check_is_fitted


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
  Parameters
  ----------
  use_idf : boolean, optional (default=True)
  k1 : float, optional (default=2.0)
  b : float, optional (default=0.75)
  References
  ----------
  Okapi BM25: a non-binary model - Introduction to Information Retrieval
  http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
  """

    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = (
            X.data
            * (self.k1 + 1)
            / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        )
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, "_idf_diag")

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError(
                    "Input has n_features=%d while the model"
                    " has been trained with n_features=%d"
                    % (n_features, expected_n_features)
                )
            # *= doesn't work
            X = X * self._idf_diag

        return X


class MyBM25Transformer(BM25Transformer):
    """
  To be used in sklearn pipeline, transformer.fit()
  needs to be able to accept a "y" argument
  """

    def fit(self, x, y=None):
        super().fit(x)


class BM25Vectorizer(feature_extraction.text.TfidfVectorizer):
    """
  Drop-in, slightly better replacement for TfidfVectorizer
  Best results if text has already gone through stopword removal and lemmatization
  """

    def __init__(self):
        self.vec = pipeline.make_pipeline(
            feature_extraction.text.CountVectorizer(binary=True), MyBM25Transformer(),
        )
        super().__init__()

    def fit(self, raw_documents, y=None):
        return self.vec.fit(raw_documents)

    def transform(self, raw_documents, copy=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return self.vec.transform(raw_documents)


class BM25:
    def fit(self, corpus, question_train, ids, question_train_ids):
        self.corpus = corpus
        self.ids = ids
        self.question_ids = question_train_ids

        # self.joined_corpus = []
        self.joined_corpus = corpus

        self.question_train = question_train
        # for fact in corpus:
        #     self.joined_corpus.append(" ".join(fact))

        self.vectorizer = BM25Vectorizer().fit(self.joined_corpus + self.question_train)
        self.vectorizer_questions = BM25Vectorizer().fit(
            self.joined_corpus + self.question_train
        )
        self.transformed_corpus = self.vectorizer.transform(self.joined_corpus)
        self.transformed_corpus_questions = self.vectorizer_questions.transform(
            self.question_train
        )

    def query(self, query, top_k):
        ordered_ids = []
        scores = []

        transformed_query = self.vectorizer.transform(query)
        TFIDF_dist = cosine_distances(transformed_query, self.transformed_corpus)
        res = []

        for index in np.argsort(TFIDF_dist)[0][:top_k]:
            t_id = self.ids[index]
            score = 1 - TFIDF_dist[0][index]
            res.append({"id": t_id, "score": score})

        return res

    def question_similarity(self, query):
        ordered_ids = []
        scores = []

        transformed_query = self.vectorizer_questions.transform(query)
        TFIDF_dist = cosine_distances(
            transformed_query, self.transformed_corpus_questions
        )
        res = []

        for index in np.argsort(TFIDF_dist)[0]:
            t_id = self.question_ids[index]
            score = 1 - TFIDF_dist[0][index]
            res.append({"id": t_id, "score": score})

        return res
