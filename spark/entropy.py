from typing import Dict

from databricks import koalas as ks
from pyspark.sql.functions import log

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


def entropy(data, base: float = 2.0, dropna: bool = True) -> Dict[str, float]:
    '''

    :param data: pyspark.sql.dataframe.DataFrame
    :param base: None or positive float
    :param dropna:
    :return:
    '''
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    kdf = ks.DataFrame(data)
    return {col: _entropy(kdf[col], base=base, dropna=dropna) for col in kdf.columns}


def _entropy(data, base: float = 2.0, dropna: bool = True) -> float:
    '''

    :param data:
    :param base:
    :param dropna:
    :return:
    '''
    probabilities = data.value_counts(normalize=True, sort=True, dropna=dropna)
    probabilities = probabilities.to_frame().to_spark()

    name = probabilities.dtypes[0][0]
    base = 2.0 if base is None else base
    df = probabilities.select(-probabilities[name] * log(base, probabilities[name]))

    kdf = ks.DataFrame(df)

    return float(kdf.sum().values[0])
