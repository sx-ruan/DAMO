# coding: utf-8
from basic import *


BASES = 'A', 'C', 'G', 'T'
PAIRS = {'A': 'T',
         'C': 'G',
         'G': 'C',
         'T': 'A',
         'N': 'N'}


class Mapper(dict):
    def __init__(self, label, mapper):
        """
        :param label: iterable
        :param mapper: dict
        :return:
        """
        self.label = tuple(label)
        super(Mapper, self).__init__(mapper)

    def __add__(self, other):
        """
        :param other: Mapper
        :return: Mapper
        """
        label = tuple(x + y for x in self.label for y in other.label)
        mapper = dict(
            (x + y, Outer(self[x], other[y])) for x in self for y in other
        )
        return Mapper(label, mapper)


class MapperWYK(Mapper):
    def __init__(self):
        super(MapperWYK, self).__init__('WYK',
            {'A': [+1, -1, -1],
             'C': [-1, +1, -1],
             'G': [-1, -1, +1],
             'T': [+1, +1, +1]}
        )


class MapperA0(Mapper):
    def __init__(self):
        super(MapperA0, self).__init__('CGT',
            {'A': [0, 0, 0],
             'C': [1, 0, 0],
             'G': [0, 1, 0],
             'T': [0, 0, 1]}
        )


class MapperC0(Mapper):
    def __init__(self):
        super(MapperC0, self).__init__('GTA',
            {'A': [0, 0, 1],
             'C': [0, 0, 0],
             'G': [1, 0, 0],
             'T': [0, 1, 0]}
        )


class MapperG0(Mapper):
    def __init__(self):
        super(MapperG0, self).__init__('TAC',
            {'A': [0, 1, 0],
             'C': [0, 0, 1],
             'G': [0, 0, 0],
             'T': [1, 0, 0]}
        )


class MapperT0(Mapper):
    def __init__(self):
        super(MapperT0, self).__init__('ACG',
            {'A': [1, 0, 0],
             'C': [0, 1, 0],
             'G': [0, 0, 1],
             'T': [0, 0, 0]}
        )


class Encoder(list):
    decoder = dict(
        A=MapperA0(), A0=MapperA0(),
        C=MapperC0(), C0=MapperC0(),
        G=MapperG0(), G0=MapperG0(),
        T=MapperT0(), T0=MapperT0(),
        WYK=MapperWYK()
    )

    def __init__(self, width, encoding, level=1):
        """
        :param width: int
        :param encoding: str
        :param level: int
        :return:
        """
        self.width = width
        self.encoding = encoding
        self.level = level

        self.sizes = np.arange(level) + 1
        """:type: ndarray"""

        base = len(BASES) - 1
        self.nlabels = base ** self.sizes  # output int
        self.npos = width - self.sizes + 1
        self.shapes = zip(self.nlabels, self.npos)
        self.m = np.multiply(self.nlabels, self.npos)  # output int
        self.n = sum(self.m)

        super(Encoder, self).__init__()
        if encoding in self.decoder:
            self.MakeMap([encoding] * width)
        else:
            assert len(encoding) == width
            self.MakeMap(list(encoding))

    def MakeMap(self, syms):
        """
        :param syms: list[str]
        :return:
        """
        for i in self.sizes:
            for keys in Subset(syms, i):
                mappers = [self.decoder[k] for k in keys]
                mapper = reduce(lambda x, y: x + y, mappers)
                self.append(mapper)

    def Encode(self, seq):
        """
        :param seq: str
        :return: list[int]
        """
        assert len(seq) == self.width
        keys = []
        for i in self.sizes:
            keys.extend(Subset(seq, i))
        code = []
        for mapper, key in zip(self, keys):
            code.extend(mapper[key])
        return code
