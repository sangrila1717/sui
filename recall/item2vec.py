from typing import Union, Tuple, List, Dict

from pyspark.sql import DataFrame
from pyspark.ml.feature import Word2Vec as SparkWord2Vec
from gensim.models.word2vec import Word2Vec as GensimWord2Vec


class Item2Vec:
    '''
    Examples:
        >>> i2v = Item2Vec(vector_size=4, min_count=1)
        >>> i2v.train(data=[['id1', 'id2', 'id3', 'id4'], ['id3', 'id2', 'id4']], epochs=5)
        >>> i2v.recall_top_n('id1', top_n=2, id_only=True)
        ['id3', 'id2']
        >>> len(i2v.vocab['id1'])
        4
        >>> i2v.recall_all_top_n(top_n=2, id_only=True)
        {'id4': ['id3', 'id2'], 'id3': ['id4', 'id2'], 'id2': ['id3', 'id4'], 'id1': ['id3', 'id2']}
        >>> type(i2v.vectors)
        <class 'dict'>
    '''
    def __init__(self, vector_size: int, window: int = 5, min_count: int = 1, workers: int = 1,
                 learning_rate: float = 0.025, max_session_length: int = 1000, framework: str = 'gensim') -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.learning_rate = learning_rate
        self.max_session_length = max_session_length
        self.epochs = 1
        self.__framework = framework.lower()
        self.__vocab = None
        self.__vectors = None
        if self.__framework == 'gensim':
            self.__model = GensimWord2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count,
                                          workers=self.workers, alpha=self.learning_rate)
        elif self.__framework == 'spark':
            self.__model = SparkWord2Vec(vectorSize=self.vector_size, windowSize=self.window, minCount=self.min_count,
                                         numPartitions=self.workers, stepSize=self.learning_rate,
                                         maxSentenceLength=self.max_session_length, maxIter=self.epochs)

    @property
    def vocab(self):
        return self.__vocab

    @property
    def vectors(self) -> Dict[str, List[float]]:
        if self.__vectors is None and self.__vocab is not None:
            self.__vectors = dict()
            for item_id in self.__vocab.index_to_key:
                self.__vectors[item_id] = list(self.__vocab[item_id])
            return self.__vectors
        elif self.__vectors is not None:
            return self.__vectors
        else:
            # There is no vocabulary built in this model
            return None

    @property
    def framework_list(self) -> Tuple[str, ...]:
        return 'gensim', 'spark'

    @property
    def framework(self) -> str:
        return self.__framework

    @framework.setter
    def framework(self, framework) -> None:
        assert framework.lower() in self.framework_list
        self.__framework = framework.lower()

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        if self.__framework == 'gensim' and isinstance(model, GensimWord2Vec):
            self.__model = model

    def train(self, data: Union[DataFrame, List[List[str]], str], epochs: int = 1) -> None:
        self.epochs = epochs
        if self.framework == 'spark':
            self.__model.setMaxIter(self.epochs)
            self.__model.fit(dataset=data)
        elif self.framework == 'gensim':
            if isinstance(data, str):
                try:
                    self.__model.build_vocab(corpus_file=data)
                except TypeError:
                    raise FileNotFoundError(
                        f"\nCannot find the file '{data}'.\n"
                        "Parameter data in string indicates the corpus file path.\n"
                        "If you want to train with a spate of sentences, please use a list of id lists as if:\n"
                        "[['item1', 'item2'], ['item2', 'item1', 'item3']]"
                    )
                else:
                    self.__model.train(corpus_file=data, total_words=len(self.__model.wv), epochs=self.epochs)
                    self.__vocab = self.__model.wv
            else:
                try:
                    self.__model.build_vocab(corpus_iterable=data)
                except TypeError as err:
                    raise err
                else:
                    self.__model.train(corpus_iterable=data, total_words=len(self.__model.wv), epochs=self.epochs)
                    self.__vocab = self.__model.wv

    def recall_top_n(self, item_id: str, top_n: int = 20, id_only: bool = False) -> List[Union[Tuple[str, float], str]]:
        if id_only:
            return [_[0] for _ in self.model.wv.most_similar(positive=item_id, topn=top_n)]
        else:
            return self.model.wv.most_similar(positive=item_id, topn=top_n)

    def recall_all_top_n(self, top_n: int = 20, id_only: bool = False) -> Dict[str, List[Tuple[str, float]]]:
        result = dict()
        for item_id in self.__vocab.index_to_key:
            result[item_id] = self.recall_top_n(item_id=item_id, top_n=top_n, id_only=id_only)
        return result

    def similarity(self, item_id, candidate_id):
        return self.model.wv.similarity(item_id, candidate_id)

    def dump(self, path: str):
        self.model.save(path)

    @staticmethod
    def restore(path: str, framework: str = 'gensim'):
        # TODO
        pass
