from typing import List, Tuple

import pandas as pd
import numpy as np


# def sliced_by_timesteps(data: pd.DataFrame, target: str, timesteps: int = 1, padding: int = None) -> Tuple[
#     List[List[List[float]]], List[List[List[float]]]]:
#     '''
#     Examples:
#         >>> data = pd.DataFrame({
#         ... 'col1': [1, 2, 3, 2, 3, 4, 3, 4, -1],
#         ... 'col2': [0, -1, 0, 1, 0, 1, 4, 2, 4],
#         ... 'col3': [6, 4, 6, 8, 6, 12, 24, 12, 24],
#         ... 'col4': [48, 48, 24, 12, 0, -12, 0, 1, 2]
#         ... })
#         >>> len(sliced_by_timesteps(data, 'col4', timesteps=3, padding=0))
#         8
#
#     '''
#     y = pd.DataFrame(data.pop[target])
#
#     if timesteps < 1:
#         timesteps = 1
#
#     if padding is None:
#         data = data.to_numpy()
#         y = y.to_numpy()
#     elif padding == 'duplicate':
#         pass
#     elif isinstance(padding, float):
#         head = np.array([[padding] * data.shape[1]] * (timesteps - 1))
#         data = np.append(head, data.to_numpy()).reshape(timesteps - 1 + data.shape[0], data.shape[1])
#         y = np.append(head, y.to_numpy()).reshape(timesteps - 1 + y.shape[0], y.shape[1])
#
#     features = []
#     labels = []
#     for i in range(data.shape[0] - timesteps):
#         features.append([list(data[i + _]) for _ in range(timesteps)])
#         labels.append([list(y[i + _]) for _ in range(timesteps)])
#
#     return features, labels


def sliced_by_timesteps(data: pd.DataFrame, timesteps: int = 1, padding: float = None) -> List[List[List[float]]]:
    '''
    Examples:
        >>> data = pd.DataFrame({
        ... 'col1': [1, 2, 3, 2, 3, 4, 3, 4, -1],
        ... 'col2': [0, -1, 0, 1, 0, 1, 4, 2, 4],
        ... 'col3': [6, 4, 6, 8, 6, 12, 24, 12, 24],
        ... 'col4': [48, 48, 24, 12, 0, -12, 0, 1, 2]
        ... })
        >>> for i in sliced_by_timesteps(data, timesteps=3, padding=0.01)[:5]:
        ...     print(i)
        [[0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [1.0, 0.0, 6.0, 48.0]]
        [[0.01, 0.01, 0.01, 0.01], [1.0, 0.0, 6.0, 48.0], [2.0, -1.0, 4.0, 48.0]]
        [[1.0, 0.0, 6.0, 48.0], [2.0, -1.0, 4.0, 48.0], [3.0, 0.0, 6.0, 24.0]]
        [[2.0, -1.0, 4.0, 48.0], [3.0, 0.0, 6.0, 24.0], [2.0, 1.0, 8.0, 12.0]]
        [[3.0, 0.0, 6.0, 24.0], [2.0, 1.0, 8.0, 12.0], [3.0, 0.0, 6.0, 0.0]]
        >>> y = data['col4']
        >>> sliced_by_timesteps(y)
        [[[48]], [[48]], [[24]], [[12]], [[0]], [[-12]], [[0]], [[1]]]

    '''
    data = pd.DataFrame(data)

    if timesteps < 1:
        timesteps = 1

    if padding is not None:
        head = np.array([[padding] * data.shape[1]] * (timesteps - 1))
        data = np.append(head, data.to_numpy()).reshape(timesteps - 1 + data.shape[0], data.shape[1])
    else:
        data = data.to_numpy()

    result = []
    for i in range(data.shape[0] - timesteps):
        result.append([list(data[i + _]) for _ in range(timesteps)])

    return result
