from tensorflow.keras.initializers import Initializer, GlorotNormal, GlorotUniform
from tensorflow.keras.losses import Loss, BinaryCrossentropy, MSE
from tensorflow.keras.optimizers import Optimizer, Adam, Ftrl

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiValueError(ValueError):
    pass


def get_init(initializer_name: str) -> Initializer:
    """
    Get an object in tensorflow.keras.initializers by name.
    :param initializer_name:
        str
        Support initializer_name without case sensitive:
            'glorotnormal'
            'glorotuniform'

    :return:
        Initializer
        An Initializer object.
    """
    initializers = {
        'glorotnormal': GlorotNormal(),
        'glorotuniform': GlorotUniform(),
    }

    initializer_name = initializer_name.strip().lower()

    try:
        return initializers[initializer_name]
    except KeyError as keyerr:
        raise SuiValueError(f'{keyerr} is not a valid initializer name.')


def get_loss(loss_name: str) -> Loss:
    """
    Get an object in tensorflow.keras.losses by name.
    :param optimizer_name:
        str
        Support loss_name without case sensitive:
            'sigmoid'
            'mse'

    :return:
        Loss
        A loss object.
    """
    losses = {
        'sigmoid': BinaryCrossentropy(),
        'mse': MSE(),
    }

    loss_name = loss_name.strip().lower()

    try:
        return losses[loss_name]
    except KeyError as keyerr:
        raise SuiValueError(f'{keyerr} is not a valid loss name.')


def get_optimizer(optimizer_name: str, learning_rate: float = 0.001) -> Optimizer:
    """
    Get an object in tensorflow.keras.optimizers by name.
    :param optimizer_name:
        str
        Support optimizer_name without case sensitive:
            'adam'
            'ftrl'

    :param learning_rate:
        float
        Learning rate.

    :return:
        Optimizer
        An optimizer object.
    """
    optimizers = {
        'adam': Adam,
        'ftrl': Ftrl,
    }

    optimizer_name = optimizer_name.strip().lower()

    try:
        return optimizers[optimizer_name](learning_rate=learning_rate)
    except KeyError as keyerr:
        raise SuiValueError(f'{keyerr} is not a valid optimizer name.')
