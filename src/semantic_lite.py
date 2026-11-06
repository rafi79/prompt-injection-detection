"""
Semantic-lite similarity stage.

A purely local semantic representation - TF-IDF + truncated SVD (LSA), fit
only on the training split - and each example's cosine similarity to
prototypes of known attack and benign training examples. No pretrained
model, no download, CPU only.

Configurable variants (all ablated in the paper):
  analyzer      "word" (word 1-2 grams, the original v2), "char" (char_wb
                2-5 grams, tolerant to obfuscation and non-English text) or
                "both" (concatenation)
  n_prototypes  1 = class centroids (original v2); k > 1 = k-means
                prototypes per class, similarity = max over prototypes
  knn           0 = off; k > 0 adds the attack fraction among the k nearest
                training examples (cosine in LSA space)

Features returned per text: [sim_pos, sim_neg, sim_pos - sim_neg]
(+ [knn_attack_fraction] when knn > 0).

Cross-fitting: when these features are used to train the downstream
logistic regression, the training rows' features must be computed
out-of-fold (see `crossfit_features`), otherwise every training row is
compared with centroids/neighbours it helped build (optimistic bias).
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


class SemanticLiteEncoder:
    def __init__(self, n_components: int = 100, max_features: int = 20000, analyzer: str = "word",
                 n_prototypes: int = 1, knn: int = 0, random_state: int = 42):
        self.n_components = n_components
        self.max_features = max_features
        self.analyzer = analyzer
        self.n_prototypes = n_prototypes
        self.knn = knn
        self.random_state = random_state
        self._fitted = False

    def clone(self):
        return SemanticLiteEncoder(self.n_components, self.max_features, self.analyzer,
                                   self.n_prototypes, self.knn, self.random_state)

    @property
    def n_features(self):
        return 3 + (1 if self.knn else 0)

    def _make_vectorizers(self):
        vs = []
        if self.analyzer in ("word", "both"):
            vs.append(TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True))
        if self.analyzer in ("char", "both"):
            vs.append(TfidfVectorizer(max_features=max(self.max_features, 30000), analyzer="char_wb",
                                      ngram_range=(2, 5), sublinear_tf=True))
        return vs

    def _tfidf(self, texts, fit=False):
        mats = [v.fit_transform(texts) if fit else v.transform(texts) for v in self.vectorizers]
        return mats[0] if len(mats) == 1 else hstack(mats).tocsr()

    def _prototypes(self, Z):
        if self.n_prototypes <= 1 or len(Z) <= self.n_prototypes:
            c = Z.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=self.n_prototypes, n_init=4, random_state=self.random_state).fit(Z)
            c = km.cluster_centers_
        return normalize(c)

    def fit(self, texts, labels):
        labels = np.asarray(labels)
        self.vectorizers = self._make_vectorizers()
        X = self._tfidf(list(texts), fit=True)
        k = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        Z = normalize(self.svd.fit_transform(X))
        # TruncatedSVD.transform(X) == X @ components_.T; keeping a contiguous
        # copy of components_.T avoids ~7 ms of per-call overhead (profiled)
        self._W = np.ascontiguousarray(self.svd.components_.T)
        self.pos_protos = self._prototypes(Z[labels == 1])
        self.neg_protos = self._prototypes(Z[labels == 0])
        # kept for backward compatibility with the original v2 attribute names
        self.pos_centroid, self.neg_centroid = self.pos_protos[0], self.neg_protos[0]
        if self.knn:
            self.train_Z, self.train_y = Z, labels
        self._fitted = True
        return self

    def embed(self, texts):
        X = self._tfidf(list(texts))
        W = getattr(self, "_W", None)
        if W is None:
            W = self._W = np.ascontiguousarray(self.svd.components_.T)
        return normalize(np.asarray(X @ W))

    def similarity_features(self, texts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SemanticLiteEncoder must be fit() first.")
        Z = self.embed(texts)
        sp = (Z @ self.pos_protos.T).max(axis=1)
        sn = (Z @ self.neg_protos.T).max(axis=1)
        cols = [sp, sn, sp - sn]
        if self.knn:
            S = Z @ self.train_Z.T
            k = min(self.knn, S.shape[1])
            nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
            cols.append(self.train_y[nn].mean(axis=1))
        return np.stack(cols, axis=1)

    def similarity_features_one(self, text: str) -> np.ndarray:
        return self.similarity_features([text])[0]


def crossfit_features(encoder: SemanticLiteEncoder, texts, labels, n_folds: int = 5, seed: int = 42):
    """Out-of-fold similarity features for the training rows."""
    texts = np.asarray(list(texts), dtype=object)
    labels = np.asarray(labels)
    out = np.zeros((len(texts), encoder.n_features))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(texts, labels):
        enc = encoder.clone().fit(texts[tr], labels[tr])
        out[te] = enc.similarity_features(texts[te])
    return out
