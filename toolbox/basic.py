# coding: utf-8
import itertools as itt
import numpy as np


def Col(array):
    """
    :param array: array_like
    :return: ndarray
    >>> Col([0, 1])
    array([[0],
           [1]])
    """
    return np.reshape(array, (-1, 1))


def Flip(dictionary):
    """
    :param dictionary: dict
    :return: dict
    >>> Flip({1: 2})
    {2: 1}
    """
    return dict((v, k) for k, v in dictionary.iteritems())


def Product(string, repeat):
    """
    :param string: str
    :param repeat: int
    :return: list[str]
    >>> Product('ab', 2)
    ['aa', 'ab', 'ba', 'bb']
    """
    return [''.join(x) for x in itt.product(string, repeat=repeat)]


def Subset(array, length):
    """
    :param array: array_like
    :param length: int
    :return: list
    >>> Subset('abcd', 3)
    ['abc', 'bcd']
    >>> Subset([1, 2], 3)
    Traceback (most recent call last):
      ...
    AssertionError
    """
    n = len(array) - length + 1
    assert n > 0
    return [array[i : i+length] for i in range(n)]


def Outer(*iterables):
    """
    :param iterables: list[iterable]
    :return: list[Number]
    >>> Outer([1, 2], [3, 4])
    [3, 4, 6, 8]
    """
    return [np.prod(x) for x in itt.product(*iterables)]


def Id(x):
    """
    :param x: object
    :return: object
    """
    return x


def Group(iterable, size=1):
    """
    :param iterable: iterable
    :param size: int
    :return: iterator
    >>> list(Group([1, 2]))
    [(1,), (2,)]
    >>> list(Group([1, 2], 2))
    [(1, 2)]
    """
    iters = [iter(iterable)] * size
    return itt.izip(*iters)


def In(x, y):
    """
    :param x: array_like
    :param y: array_like
    :return: bool
    >>> In(1, [2, 3])
    False
    >>> In(1, 1)
    True
    """
    return x in np.array(y)


def KL(x, y):
    """
    Calculate the Kullback-Leibler divergence.
    :param x: iterable
    :param y: iterable
    :return: float
    >>> KL([1, 2], [-3, 4])
    Traceback (most recent call last):
      ...
    AssertionError
    """
    p = np.fromiter(x, float)
    q = np.fromiter(y, float)
    assert min(p) > 0 and min(q) > 0
    p /= sum(p)
    q /= sum(q)
    return np.log(p / q).dot(p)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
