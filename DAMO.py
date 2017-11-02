# coding=utf-8
from helpers import *
from sklearn import metrics
import argparse


LEARNING_RATES = 1.0, 0.55, 0.1


class Mapper0(Mapper):
    def __init__(self):
        super(Mapper0, self).__init__('ACGT', {'A': [1, 0, 0, 0],
                                               'C': [0, 1, 0, 0],
                                               'G': [0, 0, 1, 0],
                                               'T': [0, 0, 0, 1]})


class Encoder2(list):
    decoder = {'0': Mapper0()}

    def __init__(self, width, encoding='0', level=1):
        """
        :param width: int
        :param encoding: str
        :param level: int
        :return:
        """
        self.hash = {}

        self.width = width
        self.encoding = encoding
        self.level = level

        self.sizes = np.arange(level) + 1
        """:type: ndarray"""

        base = len(BASES)
        self.nlabels = base ** self.sizes  # output int
        self.npos = width - self.sizes + 1
        self.shapes = zip(self.nlabels, self.npos)
        self.m = np.multiply(self.nlabels, self.npos)  # output int
        self.n = sum(self.m)

        super(Encoder2, self).__init__()
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
        if seq in self.hash:
            return self.hash[seq]
        else:
            assert len(seq) == self.width
            keys = []
            for i in self.sizes:
                keys.extend(Subset(seq, i))
            code = []
            for mapper, key in zip(self, keys):
                code.extend(mapper[key])
            self.hash[seq] = code
            return code


def read_fasta(in_fasta):
    with open(in_fasta) as f:
        for row in f:
            if not row.startswith('>'):
                yield row.strip()


def cal_auc(scored_positive, scored_negative):
    y_true = [1] * len(scored_positive) + [0] * len(scored_negative)
    y_score = scored_positive + scored_negative
    return metrics.roc_auc_score(y_true, y_score), y_true, y_score


def scoreAllSeq(seqs_list, pwm, enc):
    scores = []
    sites = []
    for seqs in seqs_list:
        max_score, max_site = get_max_site(seqs, pwm, enc)
        scores.append(max_score)
        sites.append(max_site)
    return scores, sites


def get_max_site(seqs, pwm, enc):
    """
    :param seqs: 
    :param pwm: 
    :type enc: Encoder | Encoder2
    :return: 
    """
    max_score = -np.inf
    max_site = ''
    for seq in seqs:
        score = np.dot(enc.Encode(seq), pwm)
        if score > max_score:
            max_score = score
            max_site = seq
    return max_score, max_site


def gen_pwm(seqs, level, enc):
    """
    :param seqs: 
    :param level: 
    :type enc: Encoder | Encoder2
    :rtype: np.ndarray
    """
    ppms = []
    for k in xrange(level):
        nlabel, npos = enc.shapes[k]
        pfm = np.zeros((npos, nlabel))
        labels = BASES if k == 0 else tuple(x + y for x in BASES for y in BASES)
        assert nlabel == len(labels)
        assert npos + k == len(seqs[0])
        for i in xrange(npos):
            c = Counter(seq[i:i+k+1] for seq in seqs)
            pfm[i, :] = [c[key] for key in labels]

        ppms.append(PFM.Normalize(PFM.Normalize(pfm), 1e-4))

    pwm = np.log(ppms[0]).flatten()
    if level == 1:
        return pwm
    elif level == 2:
        for i in xrange(enc.npos[1]):
            ppms[1][i, :] = np.log(ppms[1][i]) - np.log(Outer(ppms[0][i], ppms[0][i+1]))
        return np.hstack((pwm, ppms[1].flatten()))
    else:
        raise NotImplementedError


def makeAMove(y_true, y_score, sites, learning_rate, pwm, level, enc):
    ind = np.argsort(-np.asarray(y_score), kind='mergesort')  # type: np.ndarray
    flags = [y_true[i] for i in ind]
    seqs = [sites[i] for i in ind]

    idx_first_neg = flags.index(0)
    idx_last_pos = len(flags) - flags[::-1].index(1)

    pos_mixture = []
    neg_mixture = []
    for flag, seq in zip(flags, seqs)[idx_first_neg:idx_last_pos]:
        pos_mixture.append(seq) if flag == 1 else neg_mixture.append(seq)

    return pwm + learning_rate * (
        gen_pwm(pos_mixture, level, enc) - gen_pwm(neg_mixture, level, enc)
    )


def scorePerceptron(pwm, read_seq, level, enc):
    assert isinstance(read_seq, readProcessSeq)

    scored_positive, pos_sites = scoreAllSeq(read_seq.pos_splitted, pwm, enc)
    scored_negative, neg_sites = scoreAllSeq(read_seq.neg_splitted, pwm, enc)
    auc, y_true, y_score = cal_auc(scored_positive, scored_negative)

    for learning_rate in LEARNING_RATES:
        new_pwm = makeAMove(y_true, y_score, pos_sites + neg_sites, learning_rate, pwm, level, enc)

        temp_scored_positive, _ = scoreAllSeq(read_seq.pos_splitted, new_pwm, enc)
        temp_scored_negative, _ = scoreAllSeq(read_seq.neg_splitted, new_pwm, enc)
        temp_auc = cal_auc(temp_scored_positive, temp_scored_negative)[0]

        if temp_auc > auc:
            return new_pwm

    return pwm


def splittingFun(in_string, motif_length):
    seqs = Subset(in_string, motif_length)
    return tuple(set(seqs + RC(seqs)))


def splitAndGenerateList(in_array, motif_length):
    return tuple(splittingFun(seq, motif_length) for seq in in_array)


class readProcessSeq(object):
    def __init__(self, positive_file, negative_file, motif_length):
        self.positive_samples = tuple(read_fasta(positive_file))
        self.negative_samples = tuple(read_fasta(negative_file))
        self.pos_splitted = splitAndGenerateList(self.positive_samples, motif_length)
        self.neg_splitted = splitAndGenerateList(self.negative_samples, motif_length)


def damo(positive_file, negative_file, pfm_file_name, output_flag, path='.', generations=500, level=1):
    pfm = load_PPM(pfm_file_name, skipcols=2, skiprows=1)  # type: np.ndarray
    motif_length = len(pfm)
    enc = Encoder2(motif_length, level=level)
    read_seq = readProcessSeq(positive_file, negative_file, motif_length)
    temp_pwm = np.log(pfm).flatten()
    if level == 2:
        zeros = np.zeros(enc.m[1])
        temp_pwm = np.hstack((temp_pwm, zeros))

    for _ in xrange(generations):
        old_pwm = temp_pwm
        temp_pwm = scorePerceptron(temp_pwm, read_seq, level, enc)  # type: np.ndarray

        if np.array_equal(old_pwm, temp_pwm):
            break

    with open(os.path.join(path, '%s_END_PWM.txt' % output_flag), 'w') as f:
        pwm_1, pwm_2 = np.split(temp_pwm, [enc.m[0]])  # type: np.ndarray
        for label, row in zip(BASES, np.reshape(pwm_1, (enc.npos[0], enc.nlabels[0])).astype(str).T):
            f.write(label + ': ' + ' '.join(row) + '\n')

        if level == 2:
            for label, row in zip(Product(BASES, 2), np.reshape(pwm_2, (enc.npos[1], enc.nlabels[1])).astype(str).T):
                f.write(label + ': ' + ' '.join(row) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--positive', required=True, help='path of positive sequences (FASTA format)')
    parser.add_argument('-n', '--negative', required=True, help='path of negative sequences (FASTA format)')
    parser.add_argument('-s', '--seed', required=True, help='path of the initial position frequency matrix')
    parser.add_argument('-f', '--flag', default='DAMO',
                        help='prefix of the output filename (optional, default: "DAMO")')
    parser.add_argument('-g', '--generation', type=int, default=500,
                        help='number of optimization iterations (optional, default: 500)')
    parser.add_argument('-i', '--interaction', action='store_true',
                        help='consider adjacent di-nucleotide interactions (optional, default: False)')
    parser.add_argument('-o', '--output', default=os.getcwd(),
                        help='output directory (optional, default: current working directory)')
    parser.add_argument('-v', '--version', action='version', version='1.0.1')
    args = parser.parse_args()

    damo(args.positive, args.negative, args.seed, args.flag, args.output, args.generation, int(args.interaction) + 1)


if __name__ == '__main__':
    main()
