from typing import Union

from pyspark.sql import DataFrame
from pyspark.ml.feature import Word2Vec as SparkWord2Vec
from gensim.models.word2vec import Word2Vec as GensimWord2Vec
from gensim.models.word2vec import LineSentence


class Item2Vec:
    def __init__(self, vector_size: int, window: int = 5, min_count: int = 5, workers: int = 1,
                 learning_rate: float = 0.025, max_session_length: int = 1000, framework: str = 'gensim') -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.learning_rate = learning_rate
        self.max_session_length = max_session_length
        self.epochs = 1
        self.framework = framework.lower()
        if self.framework == 'gensim':
            self.model = GensimWord2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count,
                                        workers=self.workers, alpha=self.learning_rate)
            self.vocab = self.model.wv.key_to_index
        elif self.framework == 'spark':
            self.model = SparkWord2Vec(vectorSize=self.vector_size, window=self.window, minCount=self.min_count,
                                       numPartitions=self.workers, stepSize=self.learning_rate,
                                       maxSentenceLength=self.max_session_length)

    def fit(self, data: Union[DataFrame, str], epochs: int = 1) -> None:
        self.epochs = epochs
        if self.framework == 'spark':
            self.model.setMaxIter(self.epochs)
            self.model.fit(dataset=data)
        elif self.framework == 'gensim':
            vocab = LineSentence(data)
            print(vocab)
            self.model.train(corpus_file=vocab, epochs=self.epochs)
            self.vocab = self.model.wv.vocab

    def recall_top_n(self, item_id: str, top_n: int = 20):
        return self.model.wv.most_similar(positive=item_id, topn=top_n)

    def similarity(self, item_id, candidate_id):
        return self.model.wv.similarity(item_id, candidate_id)

    def dump(self, path: str):
        self.model.save(path)

    @staticmethod
    def restore(path: str, framework: str = 'gensim'):
        return GensimWord2Vec.load(path)
