# coding: utf-8
from systm import *
from codec import *
from scipy import stats


class RC(list):
    def __init__(self, iterable):
        super(RC, self).__init__(self.RC(seq) for seq in iterable)

    @classmethod
    def RC(cls, seq):
        """
        :param seq: str
        :return: str
        """
        return ''.join(PAIRS[i] for i in reversed(seq))

    @classmethod
    def IsSym(cls, seq):
        """
        :param seq: str
        :return: bool
        """
        return cls.RC(seq) == seq


class Record(object):
    def __init__(self, count=0.0):
        self.count = float(count)

    def __str__(self):
        return str(self.count)

    def __cmp__(self, other):
        return cmp(self.count, other.count)

    def __iadd__(self, other):
        self.count += other.count
        return self

    def __idiv__(self, other):
        self.count /= other.count
        return self

    def Add(self, count):
        self.count += float(count)

    def Div(self, count):
        self.count /= float(count)


class Library(dict):
    ValueClass = Record

    def __init__(self, Format=Id):
        """
        :param Format: callable
        :return:
        """
        self.Format = Format
        super(Library, self).__init__()

    def Get(self, name='count', iterable=None):
        """
        :param name: str
        :param iterable: iterable | None
        :return: list
        """
        if iterable is None:
            iterable = self.iterkeys()
        return [getattr(self[key], name) for key in iterable]

    def Sum(self):  # sum of all counts
        return sum(self.Get())

    def HasKey(self, *keys):
        """
        :param keys: list[str]
        :return: bool
        """
        return all(key in self for key in keys)

    def Encode(self, Encode):
        """
        :param Encode: Encoder.Encode
        :return:
        """
        for key, rec in self.iteritems():
            rec.code = Encode(key)

    def GetCount(self, seq, count):
        key = self.Format(seq)
        self.setdefault(key, self.ValueClass()).Add(count)

    def ReadCount(self, path, col1=0, col2=-1, mode='r',
                  start=0, stop=None, step=None):
        for line in Read(path, start, stop, step, mode):
            self.GetCount(line[col1], line[col2])
        return self

    def Export(self, iterable=None):
        if iterable is None:
            iterable = self.iterkeys()
        for key in iterable:
            yield key, self[key].count

    def Import(self, iterable, start=0, stop=None):
        for seq, count in iterable:
            self.GetCount(seq[start:stop], count)
        return self

    def Normalize(self, other):
        for key in self.keys():
            try:
                self[key] /= other[key]
            except (KeyError, ZeroDivisionError):
                del self[key]

    def Rescale(self, divisor=None):
        if divisor is None:
            divisor = self.Sum()
        for rec in self.itervalues():
            rec.Div(divisor)

    def FilterCount(self, lb=0.0, ub=np.inf):
        for key, rec in self.items():
            if not lb < rec.count < ub:
                del self[key]

    def FilterKey(self, lb=0.0, ub=1.0):
        if lb > 0.0 or lb < 1.0:
            for key in self.keys():
                freq = [key.count(x) for x in BASES]
                s = sum(freq)
                assert s == len(key)  # make sure no 'N'
                if min(freq) < lb * s or max(freq) > ub * s:
                    del self[key]

    def Tabulate(self):
        for key, rec in self.iteritems():
            yield [key] + map(str, rec.code) + [str(rec)]

    def Stats(self):
        counts = self.Get()
        tmp = [
            'Avg: %#.4g' % np.mean(counts),
            'Std: %#.4g' % np.std(counts),
            'Max: %#.4g' % max(counts),
            'Min: %#.4g' % min(counts),
            'Sum: %#.4g' % sum(counts),
            'Number: %d' % len(counts)
        ]
        print '\n'.join(tmp)

    def Bias(self, bias):
        """
        :param bias: float
        :return: ndarray
        """
        count = np.array(self.Get())
        base = count.max() / stats.gmean(count)
        power = np.log(bias) / np.log(base)
        tmp = count ** power
        return tmp / np.mean(tmp)


class WeightRecord(Record):
    def Normalize(self):
        assert min(self.weights) >= 0
        self.weights /= sum(self.weights)

    def RndWeights(self, const=None, normalize=True):
        n = len(self.contigs)
        if const is None:
            self.weights = np.ones(n)
        else:
            assert const >= 0
            self.weights = np.random.ranf(n) + const
        if normalize:
            self.Normalize()

    def SetWeights(self, weights, normalize=False):
        assert len(weights) == len(self.contigs)
        self.weights = np.array(weights, float)
        if normalize:
            self.Normalize()

    def Contig(self, key, width, both=True):
        self.contigs = Subset(key, width)
        if both:
            rc = RC(self.contigs)  # list
            self.contigs.extend(reversed(rc))  # same order as original

    def Pick(self):
        self.Normalize()
        return np.random.choice(self.contigs, p=self.weights)

    def Half(self, const=None, normalize=True):  # use only if both=True
        i = np.argmax(self.weights)
        n = len(self.contigs) / 2
        self.contigs = self.contigs[:n] if i < n else self.contigs[n:]
        self.RndWeights(const, normalize)


class ContigLibrary(Library):
    ValueClass = WeightRecord

    def NormWeights(self):
        for rec in self.itervalues():
            rec.Normalize()

    def RndWeights(self, const=None, normalize=True):
        for rec in self.itervalues():
            rec.RndWeights(const, normalize)

    def Contig(self, width, both=True):
        for key, rec in self.iteritems():
            rec.Contig(key, width, both)

    def Iter(self, name='contigs'):  # name=contigs/weights/code
        for rec in self.itervalues():
            for x in getattr(rec, name):
                yield x

    def Export(self, iterable=None):
        if iterable is None:
            iterable = self.iterkeys()
        for rec in itt.imap(self.get, iterable):
            for seq, weight in itt.izip(rec.contigs, rec.weights):
                frac = weight * rec.count
                yield seq, frac

    def New(self, start=0, stop=None, lb=0.0, ub=np.inf):
        lib = Library().Import(self.Export(), start, stop)
        lib.FilterCount(lb, ub)
        return lib


#class Prior(dict):
#    def __init__(self, *args, **kwargs):
#        dict.__init__(self, *args, **kwargs)
#        lgmean = np.log(stats.mstats.gmean(self.values()))
#        for key in self:
#            self[key] = np.log(self[key]) - lgmean
#
#    def __call__(self, seq):
#        freq = Counter(seq)
#        return np.exp(sum(freq[key] * self[key] for key in freq))
#
#class Find(object):
#    @classmethod
#    def Base(cls, iterable, Test, *args):
#        return [i for i, elem in enumerate(iterable) if Test(elem, *args)]
#
#    @classmethod
#    def Eq(cls, iterable, target):
#        return cls.Base(iterable, op.eq, target)
#
#    @classmethod
#    def Low(cls, iterable, rcond=1e-3):
#        return cls.Base(iterable, op.le, rcond * max(iterable))
#
#class Mask(object):
#    @classmethod
#    def Ind(cls, array, ind, masks):
#        for i, mask in np.broadcast(ind, masks):
#            array[i] = mask
#
#    @classmethod
#    def Rand(cls, array, size, masks):
#        ind = random.choice(len(array), size, False)
#        cls.Ind(array, ind, masks)
#
#class Cmp(object):
#    @classmethod
#    def Boolean(cls, seqX, seqY):
#        assert len(seqX) == len(seqY)
#        return [i == j for i, j in zip(seqX, seqY)]
#
#    @classmethod
#    def MatchSum(cls, seqX, seqY):
#        return sum(cls.Boolean(seqX, seqY))
#
#    @classmethod
#    def MaxLen(cls, seqX, seqY):
#        tmp = zip(seqX, seqY)
#        ind = Find.Base(tmp, lambda x: op.eq(*x))
#        ind.append(len(tmp))
#        return max(ind[0], *(np.diff(ind)-1))
#        #split = np.split(boolean, Find.Eq(boolean, False))
#        #return max(len(split[0]), *[len(x)-1 for x in split[1:]])
#
#class RandSeq(list):
#    def __init__(self, width, alphabet=SINGLE):
#        list.__init__(self, Product(alphabet, width))
#
#    def Size(self, size):
#        return random.choice(self, size, False).tolist()
#
#    def Prob(self, prob):
#        size = len(self)
#        return self.Size(int(prob * size))
#
#class Pipe(list):
#    def __init__(self, *args):
#        super(Pipe, self).__init__(args)
#
#    def __call__(self, *args):
#        assert len(self) <= len(args)
#        return [f(x) for f, x in itt.izip_longest(self, args, fillvalue=Id)]
