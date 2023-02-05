from tensorflow.keras import Sequential
from tensorflow.keras.layers import StringLookup, Embedding, SimpleRNN
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.optimizers import Adagrad

movie_id_lists = []
with open('/Users/mac/Desktop/data/movie_ids') as fin:
    for line in fin:
        movie_id_lists.append(line.strip())
movie_tensors = tf.data.Dataset.from_tensor_slices(movie_id_lists)

embedding_dimension = 32

query_model = Sequential([
    StringLookup(vocabulary=movie_id_lists, mask_token=None),
    Embedding(len(movie_id_lists) + 1, embedding_dimension),
    SimpleRNN(embedding_dimension)
])

candidate_model = Sequential([
    StringLookup(vocabulary=movie_id_lists, mask_token=None),
    Embedding(len(movie_id_lists) + 1, embedding_dimension)
])


class Model(tfrs.Model):
    def __init__(self, query_model, candidate_model, task):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model
        self.task = task

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        query_embedding = self._query_model(inputs['movies'])
        candidate_embedding = self._candidate_model(inputs['label'])
        return self.task(query_embedding, candidate_embedding, compute_metrics=not training)


record = tf.data.TFRecordDataset('/Users/mac/Desktop/data/movielens_1m.tfrecord')

feature_description = {
    'movies': tf.io.FixedLenFeature([5], tf.string),
    'label': tf.io.FixedLenFeature([1], tf.string)
}


def parse_example(example):
    feature_dict = tf.io.parse_single_example(example, feature_description)
    return feature_dict


train_ds = record.map(parse_example)
metrics = tfrs.metrics.FactorizedTopK(candidates=movie_tensors.batch(128).map(candidate_model))
task = tfrs.tasks.Retrieval(metrics)
model = Model(query_model, candidate_model, task)
model.compile(optimizer=Adagrad(learning_rate=0.1))

# (batch_size, timesteps, input_dim)
model.fit(train_ds.shuffle(100_000).batch(8192).cache(), epochs=3)
