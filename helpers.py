# coding: utf-8
import operator
import requests
import urlparse
import pickle
import linecache
import gzip
from universal import *
from scipy import random


URL = 'https://www.encodeproject.org/'


def Sym(dx):  # assume both seq & rev already in dx, and data on 1 single strand
    """
    :param dx: dict
    :return: dict
    """
    dy = {}
    for seq, rev in itt.izip(dx, RC(dx)):
        dy[seq] = 0.5 * (dx[seq] + dx[rev])
    return dy


class Distribution(dict):
    def __init__(self, pred, prefix, suffix):
        """
        :param pred: dict
        :param prefix: list[dict]
        :param suffix: list[dict]
        :return:
        """
        d = Op.Nom(pred)
        for k, v in d.items():
            it = itt.chain(self.L(k, v, prefix), self.R(k, v, suffix))
            for seq, count in it:
                assert len(seq) == len(k)
                d[seq] = d.get(seq, 0.0) + count
        super(Distribution, self).__init__(Sym(d))

    @classmethod
    def R(cls, seq, count, suffix):
        for i, d in enumerate(suffix):
            frac = seq[+i+1:]
            for k, v in d.iteritems():
                yield frac + k, v * count

    @classmethod
    def L(cls, seq, count, prefix):
        for i, d in enumerate(prefix):
            frac = seq[:-i-1]
            for k, v in d.iteritems():
                yield k + frac, v * count


def fasta(path, seqs):
    lines = []
    for i, seq in enumerate(seqs):
        lines.append(['> seq_%d' % (i + 1)])
        lines.append([seq])
    WriteCSV(path, lines)


def fmt_ppm(path, _PPM, header='', labels=BASES):
    out = [[b, '|'] + map(str, xs) for b, xs in zip(labels, np.transpose(_PPM))]
    np.savetxt(path, out, '%s', header=str(header), comments='> ')


class Freq(object):
    def __init__(self, seqs, counts):
        """
        :param seqs: iterable
        :param counts: iterable
        :return:
        """
        self.matrix = np.array(map(list, seqs))
        self.counts = np.fromiter(counts, float)
        _, self.width = self.matrix.shape

    @classmethod
    def FromIter(cls, iterable):
        """
        :param iterable: iterable
        :return: Freq
        """
        iters = list(iterable)
        seqs, counts = zip(*iters)
        return cls(seqs, counts)

    def Freq(self, ind=np.s_[:], pseudo=0.0):
        """
        :param ind: int
        :param pseudo: float
        :return: dict
        """
        freq = dict.fromkeys(BASES, pseudo)
        for chars, count in zip(self.matrix[:, ind], self.counts):
            for char in chars:
                freq[char] += count
        return freq  # not consider rc

    def GetPFM(self, pseudo=0.0):
        """
        :param pseudo: float
        :return: ndarray
        """
        dicts = [self.Freq(i, pseudo) for i in range(self.width)]
        return np.array(DictList.ToList(dicts, BASES))


def uniq_subseqs(seqs, length):
    assert 0 < length <= len(seqs[0])
    out = []
    for seq in seqs:
        subseqs = Subset(seq, length)
        out.extend(subseqs + RC(subseqs))
    return list(set(out))


def mean_std_fmt(array, n=3):
    fmt = '%.{0}f (%.{0}f)'.format(n)
    a = np.array(array, float).ravel()
    return fmt % (np.mean(a), np.std(a))


def gen_pred(seqs, _PPM):
    log_PPM = np.log(_PPM)
    n1, n2 = log_PPM.shape
    assert n2 == len(BASES)
    dicts = DictList.ToDict(log_PPM, BASES)
    pred = {}
    for key in set(seqs):
        assert n1 == len(key)
        pred[key] = np.exp(sum(d[b] for d, b in zip(dicts, key)))
    return pred


def Gradient(f, x0, h=1e-6):
    n = len(x0)
    g = np.zeros(n)
    for i in range(n):
        x = np.array(x0)
        x[i] -= h
        g[i] -= f(x)
        x = np.array(x0)
        x[i] += h
        g[i] += f(x)
    g /= 2 * h
    return g


def Hessian(f, x0, h=1e-6):
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        x = np.array(x0)
        x[i] -= h
        H[:, i] -= Gradient(f, x, h)
        x = np.array(x0)
        x[i] += h
        H[:, i] += Gradient(f, x, h)
    H /= 2 * h
    return (H + H.T) / 2


class Peaks(list):
    def sorted(self, size=None, step=1, reverse=True, rand=False):
        """
        :param size: int | None
        :param step: int
        :param reverse: bool
        :param rand: bool
        :return: list[Peak]
        """
        if rand:
            out = random.choice(self, size=size, replace=False)
        else:
            sort_pks = sorted(self, reverse=reverse)
            if size is None:
                stop = None
            else:
                stop = step * size
                assert stop <= len(self)
            out = sort_pks[0:stop:step]
        return Peaks(out)

    def get_attr(self, ind=None, name='signal'):
        if ind is None:
            ind = range(len(self))
        return [getattr(self[i], name) for i in ind]


class BedReader(Peaks):
    hd = 'chrom', 'start', 'stop', 'name', 'score', 'strand', 'signal', 'p', 'q'
    selector = {'bed broadPeak': hd, 'bed narrowPeak': hd + ('peak',)}

    def __init__(self, path, file_type, sep='\t'):
        super(BedReader, self).__init__()
        keys = self.selector[file_type]

        with gzip.open(path) as f:  # assume gzip file
            for line in f:
                values = line.rstrip().split(sep)
                assert len(keys) == len(values)
                kwargs = dict(zip(keys, values))
                self.append(Peak(**kwargs))


class Peak(object):
    def __init__(self, chrom, start, stop, signal, peak=None, **kwargs):
        self.chrom = chrom
        self.start = int(start)
        self.stop = int(stop)
        self.signal = float(signal)  # enrichment
        if peak is None or int(peak) == -1:  # Use -1 if no point-source called.
            self.peak = (self.start + self.stop) / 2  # int
        else:
            self.peak = self.start + int(peak)
            assert self.peak <= self.stop
        del kwargs

    def __cmp__(self, other):
        return cmp(self.signal, other.signal)

    def chr_path(self, path_fmt):
        return path_fmt % self.chrom

    def coord(self, shift, length):
        if length:
            half = length / 2
            mid = self.peak + shift
            return mid - half, mid + half
        else:
            return self.start + shift, self.stop + shift

    def seek(self, shift, length, path_fmt, skip=1):
        start, stop = self.coord(shift, length)
        return seek_seq(self.chr_path(path_fmt), start, stop - start, skip)


class GetScore(list):
    def __init__(self, seqs, pred_dict, use_best_site=True):
        """
        :param seqs: list[str]
        :param pred_dict: dict
        :return:
        """
        super(GetScore, self).__init__()
        width = len(next(pred_dict.iterkeys()))

        self.ind = []
        self.dists = []
        for seq in seqs:
            contigs = Subset(seq, width)
            contigs.extend(RC(contigs))

            scores = [pred_dict[key] for key in contigs]  # len(key) == width
            max_score = max(scores)
            if use_best_site:
                self.append(max_score)
            else:
                self.append(np.mean(scores))

            ind = [i for i, x in enumerate(scores) if x == max_score]
            idx = random.choice(ind)
            self.ind.append(idx)

            n = len(scores) / 2  # binding sites positions #
            self.dists.append(idx - n/2 if idx < n else idx - n - n/2)


class DataFile(dict):
    def __init__(self, *args, **kwargs):
        super(DataFile, self).__init__(*args, **kwargs)

        self.accession = self['accession']
        self.href = self['href']  # relative download path
        self.file_type = self['file_type']

        self.basename = os.path.basename(self.href)
        self.url = urlparse.urljoin(URL, self.href)
        self.response = None

    def get_response(self):
        if self.response is None:
            self.response = requests.get(self.url)

    def download(self, local_dir, mode='wb'):
        self.local_path = os.path.join(local_dir, self.basename)
        if not os.path.isfile(self.local_path):
            self.get_response()
            with open(self.local_path, mode) as f:
                f.write(self.response.content)  # binary


class Assay(dict):
    def __init__(self, *args, **kwargs):
        super(Assay, self).__init__(*args, **kwargs)
        self.accession = self['accession']
        self.files = [DataFile(d) for d in self['files']]
        self.target = Target(self['target'])

    def get_types(self, file_types, assembly):
        out = []
        for f in self.files:
            if f.file_type in file_types and f['assembly'] == assembly:
                out.append(f)
        return out


class Target(dict):
    def __init__(self, *args, **kwargs):
        super(Target, self).__init__(*args, **kwargs)
        self.gene_name = self['gene_name']

    def is_gene(self, gene_name):
        """
        :param gene_name: str
        :return: bool
        """
        return self.gene_name.upper() == gene_name.upper()


class Cmp(list):
    msg = 'r2 = %.3f, norm = %#.4g, kl = %.3f, kl_sym = %.3f (n = %d)'

    def __init__(self, x, y, normed=True):
        """
        :param x: array_like
        :param y: array_like
        :return:
        """
        a = np.array(x, float).flatten()
        b = np.array(y, float).flatten()

        assert len(a) == len(b)
        assert min(a) > 0 and min(b) > 0

        if normed:
            a /= sum(a)
            b /= sum(b)

        super(Cmp, self).__init__(self.Stats(a, b))

    @classmethod
    def FromDict(cls, dx, dy, normed=True):
        """
        :param dx: dict
        :param dy: dict
        :param normed: bool
        :return: Cmp
        """
        keys = set(dx) & set(dy)
        x = [dx[k] for k in keys]
        y = [dy[k] for k in keys]
        return cls(x, y, normed)

    @classmethod
    def Stats(cls, x, y):
        n = len(x)

        r2 = stats.pearsonr(x, y)[0] ** 2
        norm = np.fabs(x - y).sum() / n
        kl = KL(x, y)
        kl_sym = (KL(x, y) + KL(y, x)) / 2

        tup = r2, norm, kl, kl_sym, n
        print cls.msg % tup
        return tup

    def Write(self, path, sep='\t'):
        Write(path, sep.join(map(str, self)))


def get_ecdf(array, reverse=False):
    """
    Generate the empirical distribution function.
    :param array: array_like
    :param reverse: bool
    :return: float -> float
    """
    n = len(array)
    op = operator.ge if reverse else operator.le

    def ecdf(t):
        m = sum(op(x, t) for x in array)  # x <= t or x >= t if reverse
        return float(m) / float(n)

    return ecdf  # return func


def seek_seq(path, start, length, skip=1):
    width = len(getline(path, skip + 1))

    idx = start - 1
    row_idx = idx / width
    col_idx = idx % width

    lineno = row_idx + skip + 1
    line = getline(path, lineno)
    chars = list(line[col_idx:])

    while len(chars) < length:
        lineno += 1
        line = getline(path, lineno)
        if line:
            chars.extend(line)
        else:
            break

    return ''.join(chars[:length]).upper()


def getline(path, lineno):
    return linecache.getline(path, lineno).rstrip()


def load_PPM(path, skipcols=1, skiprows=0):
    return np.loadtxt(path, str, skiprows=skiprows).T[skipcols:].astype(float)


def Dump(path, obj):
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def pad_PPM(_PPM, width):
    if width > len(_PPM):
        diff = width - len(_PPM)
        prefix = diff / 2
        suffix = diff - prefix
        return np.pad(_PPM, ((prefix, suffix), (0, 0)), 'constant',
                      constant_values=(0.25,))
    else:
        return _PPM
