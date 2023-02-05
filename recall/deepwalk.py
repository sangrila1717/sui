"""DeepWalk class to initiate, train, and use a deepwalk model
described in https://arxiv.org/pdf/1403.6652.pdf
based on networkx and gensim

Date: 07/Mar/2021
Author: Li Tang
"""
from typing import Union, List, Tuple
from random import shuffle
from multiprocessing import Pool

import networkx as nx
from gensim.models import Word2Vec

from ..toolbox import random_walk

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.11'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiDeepWalkError(Exception):
    pass


class DeepWalk:
    """Initiate, train and use a deepwalk model described in https://arxiv.org/pdf/1403.6652.pdf;
    this class is implemented based on gensim and networkx.

    """

    def __init__(self, G=nx.Graph(), model=None, walk_path: list = []):
        self.G = G  # networkx graph to generate the walk path to train the embedding model
        self.model = model  # gensim word2vec model to embed node vectors
        self.walk_path = walk_path  # walk path to train the model as 'sentences'

    def add_node(self, node_for_adding):
        self.G.add_node(node_for_adding=node_for_adding)

    def add_nodes_from(self, nodes_for_adding):
        self.G.add_nodes_from(nodes_for_adding=nodes_for_adding)

    def add_edge(self, u_of_edge, v_of_edge):
        self.G.add_edge(u_of_edge=u_of_edge, v_of_edge=v_of_edge)

    def add_edges_from(self, ebunch_to_add):
        self.G.add_edges_from(ebunch_to_add=ebunch_to_add)

    def concurrent_walk(self, walk_depth: int, walking_workers: int = 4, nodes_list: List[list] = None) -> List[list]:
        """Function to obtain the series vertex data by concurrent walking in the graph.

        Args:
        walk_depth: the length for each walking result.
        walking_workers: the maximum number of processes in the pool.
        nodes_list: the list containing existing walking history;
            the last element in this list would be considered as the start for walking;
            if the nodes_list is None, every vertex in G.nodes will combine the nodes_list.

        Returns:
            a list including lists of walking series which consist of passed nodes.

        Examples:
            >>> G = nx.Graph()
            >>> G.add_nodes_from([
            ...     ('Beijing', {'Country': 'CN'}),
            ...     ('London', {'Country': 'UK'}),
            ...     ('Zhuzhou', {'Country': 'CN'}),
            ...     ('Manchester', {'Country': 'UK'}),
            ...     ('New York', {'Country': 'US'}),
            ...     ('Michigan', {'Country': 'US'}),
            ...     ('Shanghai', {'Country': 'CN'})
            ... ])
            >>> G.add_edges_from([
            ...     ('Beijing', 'Zhuzhou'),
            ...     ('Beijing', 'London'),
            ...     ('Beijing', 'New York'),
            ...     ('Beijing', 'Shanghai'),
            ...     ('Zhuzhou', 'Shanghai'),
            ...     ('Shanghai', 'London'),
            ...     ('Shanghai', 'New York'),
            ...     ('Shanghai', 'Michigan'),
            ...     ('London', 'Manchester'),
            ...     ('London', 'New York'),
            ...     ('New York', 'Michigan')
            ... ])
            >>> walker = DeepWalk(G=G)
            >>> result_list = walker.concurrent_walk(5, 2)
            >>> [_[0] for _ in result_list]
            ['Beijing', 'London', 'Zhuzhou', 'Manchester', 'New York', 'Michigan', 'Shanghai']
            >>> new_result_list = walker.concurrent_walk(8, 2, nodes_list=result_list)
            >>> len(new_result_list), len(new_result_list[0])
            (7, 8)
            >>> [result_list[i][:5] == new_result_list[i][:5] for i in range(len(result_list))]
            [True, True, True, True, True, True, True]

        """
        # if there is no specific nodes list to generate the walk path
        # uses all nodes in the graph directly
        if nodes_list is None:
            if len(self.G.nodes) == 0:
                raise ValueError("Parameter 'nodes_list' cannot be None if the length of G.nodes is 0.")

            nodes_list = [[node] for node in self.G.nodes]

        shuffle(nodes_list)

        # multiprocess pool to schedule the concurrent random walk
        pool = Pool(walking_workers)
        result = []

        for walk_path in nodes_list:
            result.append(pool.apply_async(random_walk, args=(self.G, walk_depth, walk_path)))

        pool.close()
        pool.join()

        return [_.get() for _ in result]

    def _embedding(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                   max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                   sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
                   trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(),
                   max_final_vocab=None):
        """Function to train the word2vec model based on gensim.
        Args docstring are copied from gensim.models.Word2Vec.

        Args:
        sentences : iterable of iterables, optional
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            See also the `tutorial on data streaming in Python
            <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).
        size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs : {0, 1}, optional
            If 1, hierarchical softmax will be used for model training.
            If 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {0, 1}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        max_final_vocab : int, optional
            Limits the vocab to a target vocab size by automatically picking a matching min_count. If the specified
            min_count is more than the calculated min_count, the specified min_count will be used.
            Set to `None` if not required.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the
            model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.
        sorted_vocab : {0, 1}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
            See :meth:`~gensim.models.word2vec.Word2VecVocab.sort_vocab()`.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        compute_loss: bool, optional
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.word2vec.Word2Vec.get_latest_training_loss`.
        callbacks : iterable of :class:`~gensim.models.callbacks.CallbackAny2Vec`, optional
            Sequence of callbacks to be executed at specific stages during training.

        Returns:
            None

        Examples:
            >>> walker = DeepWalk()
            >>> result_list = [
            ...     ['Beijing', 'New York', 'Beijing', 'Shanghai', 'New York'],
            ...     ['London', 'Shanghai', 'Michigan', 'New York', 'Michigan'],
            ...     ['Zhuzhou', 'Beijing', 'New York', 'Shanghai', 'New York'],
            ...     ['Manchester', 'London', 'New York', 'Shanghai', 'Beijing'],
            ...     ['New York', 'Michigan', 'Shanghai', 'Zhuzhou', 'Shanghai'],
            ...     ['Michigan', 'New York', 'Beijing', 'New York', 'London'],
            ...     ['Shanghai', 'London', 'Manchester', 'London', 'Beijing']
            ... ]
            >>> walker._embedding(sentences=result_list, size=2, min_count=1, epochs=10000)
            >>> len(walker.model['Beijing'])
            2
        """

        self.model = Word2Vec(sentences=sentences, size=size, alpha=alpha, window=window, min_count=min_count,
                              max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers,
                              min_alpha=min_alpha,
                              sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean,
                              hashfxn=hashfxn, iter=epochs, null_word=null_word,
                              trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words,
                              compute_loss=compute_loss, callbacks=callbacks,
                              max_final_vocab=max_final_vocab)

    def train(self, walk_depth: int, walking_workers: int = 4, nodes_list: List[list] = None, size: int = 100,
              alpha: float = 0.025, window: int = 5, min_count: int = 1, max_vocab_size: int = None, sample=1e-3,
              seed=1, embedding_workers: int = 3, min_alpha: float = 0.0001, sg=1, hs=0, negative=5, ns_exponent=0.75,
              cbow_mean=1, hashfxn=hash, epochs: int = 5, null_word=0, trim_rule=None, sorted_vocab=1,
              batch_words: int = 10000, compute_loss=False, callbacks=(), max_final_vocab=None):
        """Function to train the DeepWalk model

        Args:
            See also docstring in concurrent_walk and _embedding functions.

        Returns:
            None

        Examples:
            >>> G = nx.Graph()
            >>> G.add_nodes_from([
            ...     ('Beijing', {'Country': 'CN'}),
            ...     ('London', {'Country': 'UK'}),
            ...     ('Zhuzhou', {'Country': 'CN'}),
            ...     ('Manchester', {'Country': 'UK'}),
            ...     ('New York', {'Country': 'US'}),
            ...     ('Michigan', {'Country': 'US'}),
            ...     ('Shanghai', {'Country': 'CN'})
            ... ])
            >>> G.add_edges_from([
            ...     ('Beijing', 'Zhuzhou'),
            ...     ('Beijing', 'London'),
            ...     ('Beijing', 'New York'),
            ...     ('Beijing', 'Shanghai'),
            ...     ('Zhuzhou', 'Shanghai'),
            ...     ('Shanghai', 'London'),
            ...     ('Shanghai', 'New York'),
            ...     ('Shanghai', 'Michigan'),
            ...     ('London', 'Manchester'),
            ...     ('London', 'New York'),
            ...     ('New York', 'Michigan')
            ... ])
            >>> walker = DeepWalk(G=G)
            >>> walker.train(walk_depth=5, size=2)
            >>> type(walker.model['Beijing']), len(walker.model['Zhuzhou'])
            (<class 'numpy.ndarray'>, 2)

        """
        # generate the walk path first
        self.walk_path = self.concurrent_walk(walk_depth=walk_depth, walking_workers=walking_workers,
                                              nodes_list=nodes_list)
        # then the model uses the walk path as 'sentences' to train a word2vec model
        self._embedding(sentences=self.walk_path, size=size, alpha=alpha, window=window, min_count=min_count,
                        max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=embedding_workers,
                        min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent,
                        cbow_mean=cbow_mean, hashfxn=hashfxn, epochs=epochs, null_word=null_word, trim_rule=trim_rule,
                        sorted_vocab=sorted_vocab, batch_words=batch_words, compute_loss=compute_loss,
                        callbacks=callbacks, max_final_vocab=max_final_vocab)

    def update_model(self, walk_path=None, walk_path_file=None, total_examples=None, total_words=None, epochs=None,
                     start_alpha=None, end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0,
                     compute_loss=False, callbacks=()):
        """Function to update the word2vec model based on new walk path input.
        Args docstring is copied from gensim.models.Word2Vec

        Args:
        sentences : iterable of list of str
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            See also the `tutorial on data streaming in Python
            <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to`train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to `train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        word_count : int, optional
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.
        compute_loss: bool, optional
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.word2vec.Word2Vec.get_latest_training_loss`.
        callbacks : iterable of :class:`~gensim.models.callbacks.CallbackAny2Vec`, optional
            Sequence of callbacks to be executed at specific stages during training.

        Returns:
            None

        """
        self.model.train(sentences=walk_path, corpus_file=walk_path_file, total_examples=total_examples,
                         total_words=total_words, epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha,
                         word_count=word_count, queue_factor=queue_factor, report_delay=report_delay,
                         compute_loss=compute_loss, callbacks=callbacks)

    def vector(self, nodes: Union[List[str or int or float], Tuple[str or int or float], str, int, float],
               return_type: str = 'list') -> Union[list, dict]:
        """Function to return vectors of nodes in this model.

        Args:
            nodes: all nodes going to fetch the vector from this model
            return_type: can be either 'list' or 'dict' to decide the type of the returned result

        Returns:
            A list or dictionary including vectors for all these input nodes.

        Examples:
            >>> G = nx.Graph()
            >>> G.add_nodes_from([
            ...     ('Beijing', {'Country': 'CN'}),
            ...     ('London', {'Country': 'UK'}),
            ...     ('Zhuzhou', {'Country': 'CN'}),
            ...     ('Manchester', {'Country': 'UK'}),
            ...     ('New York', {'Country': 'US'}),
            ...     ('Michigan', {'Country': 'US'}),
            ...     ('Shanghai', {'Country': 'CN'})
            ... ])
            >>> G.add_edges_from([
            ...     ('Beijing', 'Zhuzhou'),
            ...     ('Beijing', 'London'),
            ...     ('Beijing', 'New York'),
            ...     ('Beijing', 'Shanghai'),
            ...     ('Zhuzhou', 'Shanghai'),
            ...     ('Shanghai', 'London'),
            ...     ('Shanghai', 'New York'),
            ...     ('Shanghai', 'Michigan'),
            ...     ('London', 'Manchester'),
            ...     ('London', 'New York'),
            ...     ('New York', 'Michigan')
            ... ])
            >>> walker = DeepWalk(G=G)
            >>> walker.train(walk_depth=5, size=2)
            >>> list_result = walker.vector('Beijing')
            >>> type(list_result), len(list_result)
            (<class 'list'>, 2)
            >>> dict_result = walker.vector(('Beijing', 'Zhuzhou'), return_type='dict')
            >>> dict_result.keys()
            dict_keys(['Beijing', 'Zhuzhou'])

        """
        # 'return_type' can be either 'list' or 'dict'
        assert return_type in ('list', 'dict'), SuiDeepWalkError(
            "The 'return_type' can be either 'list' or 'dict', got %s." % return_type)

        # if the 'nodes' is a list or tuple
        if isinstance(nodes, (list, tuple)):
            if return_type == 'list':
                result = []
                for node in nodes:
                    try:
                        # fetch the vector of this node from the model if it exists
                        result.append(list(self.model[node]))
                    except KeyError as keyerr:
                        raise SuiDeepWalkError(keyerr)
            else:
                result = {}
                for node in nodes:
                    try:
                        # fetch the vector of this node from the model if it exists
                        result.setdefault(node, list(self.model[node]))
                    except KeyError as keyerr:
                        raise SuiDeepWalkError(keyerr)
            return result

        else:
            try:
                # fetch the vector of this node from the model if it exists
                return list(self.model[nodes]) if return_type == 'list' else {nodes: list(self.model[nodes])}
            except KeyError as keyerr:
                raise SuiDeepWalkError(keyerr)

    def nearest_k(self, nodes_list: Union[list, tuple] = None, k: int = 30, negative: Union[list, tuple] = None,
                  restrict_vocab=None, indexer=None) -> dict:
        """Function to return top k nearest neighbours for every node in the input nodes_list.

        """
        if nodes_list is None:
            if len(self.G.nodes) == 0:
                raise ValueError("Parameter 'nodes_list' cannot be None if the length of G.nodes is 0.")

            nodes_list = self.G.nodes

        result = {}
        for node in nodes_list:
            result.setdefault(node, self.model.wv.most_similar(positive=node, negative=negative, topn=k,
                                                               restrict_vocab=restrict_vocab, indexer=indexer))
        return result
