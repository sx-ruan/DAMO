# coding: utf-8
import sys
from toolbox import *
from scipy import optimize
from collections import Counter


class PFM(object):
    @classmethod
    def Consensus(cls, _PFM, labels=BASES):
        ind = [np.argmax(row) for row in _PFM]
        return ''.join(labels[i] for i in ind)

    @classmethod
    def Normalize(cls, _PFM, pseudo=0.0):
        M = np.array(_PFM, float) + pseudo
        return M / np.sum(M, axis=1, keepdims=True)

    @classmethod
    def Entropy(cls, _PFM, pseudo=0.0):
        _PPM = cls.Normalize(_PFM, pseudo)
        return -np.sum(_PPM * np.log2(_PPM), axis=1)

    @classmethod
    def target_mean_IC(cls, PWM, mean_IC, pseudo=1e-8):
        assert 0 < mean_IC < 2
        PWM -= np.max(PWM, axis=1, keepdims=True)

        def obj(x):
            new_PPM = cls.PWM_PPM(x * PWM)
            return 2 - np.mean(cls.Entropy(new_PPM, pseudo)) - mean_IC

        xf = optimize.brentq(obj, 1e-3, 1e3)
        return xf * PWM

    @classmethod
    def PWM_PPM(cls, PWM, pseudo=0.0):
        return cls.Normalize(np.exp(PWM), pseudo)

    @classmethod
    def PFM_PWM(cls, _PFM, pseudo=0.0):
        return np.log(cls.Normalize(_PFM, pseudo))

    @classmethod
    def Expansion(cls, pwm, consensus):
        width = len(consensus)
        rows = np.reshape(pwm, (width, -1))

        PWM = []
        for base, row in zip(consensus, rows):
            mapper = Encoder.decoder[base]
            mat = map(mapper.get, BASES)  # a 4-by-3 matrix
            PWM.append(np.dot(mat, row))  # np.dot gives 1d array
        return PWM

    @classmethod
    def Shrink(cls, _PFM, consensus):
        if consensus in Encoder.decoder:
            consensus = [consensus] * len(_PFM)

        PWM = cls.PFM_PWM(_PFM)
        pwm = []
        for encoding, row in zip(consensus, PWM):
            if encoding != 'WYK':
                d = dict(zip(BASES, row))
                row -= d[encoding.rstrip('0')]  # subtract encoding energy
            mapper = Encoder.decoder[encoding]
            mat = map(mapper.get, BASES)
            row2 = np.dot(row, mat)
            if encoding == 'WYK':
                row2 /= 4
            pwm.extend(row2)
        return pwm

    @classmethod
    def Slice(cls, _PFM, width=None, pseudo=0.0):
        _PPM = cls.Normalize(_PFM, pseudo)
        if width is None or width == len(_PPM):
            return _PPM
        else:
            assert 0 < width < len(_PPM)
            ents = cls.Entropy(_PFM)
            sums = [sum(x) for x in Subset(ents, width)]  # list of total ent
            i = np.argmin(sums)
            return _PPM[i:i+width]  # sub-pwm of min ent (max info content)

    @classmethod
    def Predictor(cls, _PFM, width=None, labels=BASES, pseudo=0.0):
        _PPM = cls.Slice(_PFM, width, pseudo)
        dicts = DictList.ToDict(_PPM, labels)
        return Suffix(dicts).dict, _PPM


class Affix(list):
    def __init__(self, dicts):
        """
        :param dicts: list[dict]
        :return:
        """
        super(Affix, self).__init__()
        it = itt.imap(Op.Nom, dicts)

        self.dict = next(it)
        self.append(self.dict)
        for d in it:
            self.dict = self.Extended(d)
            self.append(self.dict)

    def Extended(self, d):
        """
        :param d: dict
        :return:
        """
        raise NotImplementedError


class Suffix(Affix):
    def Extended(self, d):
        return Op.Add(self.dict, d)


class Prefix(Affix):
    def Extended(self, d):
        return Op.Add(d, self.dict)


class CmpSeq(object):
    @classmethod
    def Boolean(cls, seqX, seqY):
        assert len(seqX) == len(seqY)
        return [i != j for i, j in zip(seqX, seqY)]

    @classmethod
    def Mismatch(cls, seqX, seqY):
        return sum(cls.Boolean(seqX, seqY))


class Op(object):
    @classmethod
    def Add(cls, dx, dy):
        """
        :param dx: dict
        :param dy: dict
        :return: dict
        """
        return dict((x + y, dx[x] * dy[y]) for x in dx for y in dy)

    @classmethod
    def Div(cls, dx, dy):
        """
        :param dx: dict
        :param dy: dict
        :return: dict
        """
        return dict((k, dx[k] / dy[k]) for k in set(dx) & set(dy))

    @classmethod
    def Nom(cls, d):
        """
        :param d: dict
        :return: dict
        """
        vals = d.values()
        assert min(vals) >= 0
        s = sum(vals)
        return dict((k, v / s) for k, v in d.iteritems())


class DictList(object):
    @classmethod
    def ToList(cls, dicts, keys):
        return [[d[k] for k in keys] for d in dicts]

    @classmethod
    def ToDict(cls, lists, keys):
        return [dict(itt.izip(keys, x)) for x in lists]


def hamming1(consensus, squeeze=True):
    out = []
    chars = list(consensus)
    for i, char in enumerate(chars):
        row = []
        for b in BASES:
            chars[i] = b
            row.append(''.join(chars))
        out.append(row)
        chars[i] = char

    return sorted(set(itt.chain(*out))) if squeeze else out


def max_subseq(a, b, lim):
    na = len(a)
    nb = len(b)
    n_min = min(na, nb)
    assert 0 < lim <= n_min
    for i in range(lim):
        if a[na-n_min+i:] == b[:n_min-i] or a[:n_min-i] == b[nb-n_min+i:]:
            return i
    return sys.maxint


class Quality(object):
    def __init__(self, p, shift=33):  # ASCII encoded starting with '!'(33)
        """
        :param p: float
        :param shift: int
        :return:
        """
        q = 10 * np.log10(1 / p)
        self.min = int(q + shift)

    def __call__(self, string):
        """
        :param string: str
        :return: bool
        """
        return min(map(ord, string)) >= self.min


class Formatter(list):
    def __init__(self, infile, p=0.01):
        """
        :param infile: str
        :param p: float
        :return:
        """
        super(Formatter, self).__init__()
        self.infile = infile

        with open(infile) as f:
            it = itt.imap(str.strip, f)

            if infile.endswith('.fasta'):
                for _, seq in Group(it, 2):
                    if set(seq).issubset(BASES):
                        self.append(seq)
            elif infile.endswith('.fastq'):
                qual = Quality(p)
                for key, seq, _, phred in Group(it, 4):
                    if qual(phred) and set(seq).issubset(BASES):
                        self.append(seq)
            else:
                print 'The file extension must be .fasta or .fastq\n'
                sys.exit()

    def CSV(self, path=None):
        if path is None:
            path = self.infile + '.txt'
        WriteCSV(path, Counter(self).iteritems())
