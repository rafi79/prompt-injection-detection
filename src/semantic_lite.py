"""
Semantic-lite similarity stage.

Motivation: the ablation/limitations analysis showed the lexical pipeline's
main weakness is recall on paraphrased/role-play jailbreaks that don't use
canonical trigger phrases (e.g. long persona-definition prompts). A full
transformer embedding would fix this but breaks the "lightweight, CPU-only,
no pretrained download" constraint this paper argues for.

This module instead builds a purely local semantic representation - TF-IDF
+ truncated SVD (LSA), fit only on the training split - and computes each
example's cosine similarity to the centroid of known positive (attack) and
negative (benign) training examples. This adds a soft "does this look like
known attacks in meaning, not just wording" signal at a few hundred
microseconds of extra latency, with zero external downloads.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class SemanticLiteEncoder:
    def __init__(self, n_components: int = 100, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.pos_centroid = None
        self.neg_centroid = None
        self._fitted = False

    def fit(self, texts, labels):
        X_tfidf = self.vectorizer.fit_transform(texts)
        X_svd = self.svd.fit_transform(X_tfidf)
        X_svd = normalize(X_svd)
        labels = np.array(labels)
        self.pos_centroid = X_svd[labels == 1].mean(axis=0)
        self.neg_centroid = X_svd[labels == 0].mean(axis=0)
        self.pos_centroid /= (np.linalg.norm(self.pos_centroid) + 1e-9)
        self.neg_centroid /= (np.linalg.norm(self.neg_centroid) + 1e-9)
        self._fitted = True
        return self

    def embed(self, texts):
        X_tfidf = self.vectorizer.transform(texts)
        X_svd = self.svd.transform(X_tfidf)
        return normalize(X_svd)

    def similarity_features(self, texts) -> np.ndarray:
        """Returns [sim_to_positive_centroid, sim_to_negative_centroid, margin] per text."""
        if not self._fitted:
            raise RuntimeError("SemanticLiteEncoder must be fit() first.")
        X = self.embed(texts)
        sim_pos = X @ self.pos_centroid
        sim_neg = X @ self.neg_centroid
        margin = sim_pos - sim_neg
        return np.stack([sim_pos, sim_neg, margin], axis=1)

    def similarity_features_one(self, text: str) -> np.ndarray:
        return self.similarity_features([text])[0]
