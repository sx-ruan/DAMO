# coding: utf-8
import os
import time
import csv
import itertools as itt


def CheckDir(*args):
    paths = map(os.path.dirname, args)
    for path in itt.ifilter(None, paths):
        if not os.path.exists(path):
            os.makedirs(path)


def Read(path, start=0, stop=None, step=None, mode='r'):
    data = []
    with open(path, mode) as f:
        for line in itt.islice(f, start, stop, step):
            data.append(line.split())
    print '>>\nRead %s:\n%d line(s)\n' % (path, len(data))
    return data


def Write(path, text='', mode='w'):
    with open(path, mode) as f:
        f.write(str(text))


def WriteCSV(path, text, header=None, mode='wb'):
    with open(path, mode) as f:
        writer = csv.writer(f, dialect='excel-tab')
        if header:
            writer.writerow(header)
        writer.writerows(text)


class Timer(object):
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return time.time() - self.t

    def Reset(self):
        self.t = time.time()


class File(object):
    def __init__(self, name, ext, stamped=True):
        """
        :param name: str
        :param ext: str
        :param stamped: bool
        :return:
        """
        self.name = name
        self.ext = ext.lstrip('.')
        self.stamped = stamped

    def Make(self, *args):
        return '_'.join([self.name] + list(args)) + '.' + self.ext


class Path(dict):
    def __init__(self, folder, *files):
        self.folder = folder
        self.parent = None
        self.stamps = ()
        super(Path, self).__init__((f.name, f) for f in files)

    def __div__(self, other):
        other.parent = self
        return other

    def __str__(self):
        return '\n'.join(sorted(self[key] for key in self)) + '\n'

    def __getitem__(self, key):
        f = super(Path, self).__getitem__(key)
        args = self.stamps if f.stamped else ()
        base = f.Make(*args)
        return os.path.join(self.GetPath(), base)

    def Stamp(self, *args):
        self.stamps = args

    def AddFile(self, *files):
        self.update((f.name, f) for f in files)

    def GetPath(self):
        this = self
        path = os.path.join(self.folder, '')
        while this.parent is not None:
            this = this.parent
            path = os.path.join(this.folder, path)
        return path


class Text(list):
    def __init__(self):
        self.stack = []
        self.end = None
        super(Text, self).__init__()

    def __str__(self):
        return ''.join(self)

    def __enter__(self):
        self.stack.append(self.end)
        self.end = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.append(self.stack.pop())

    def _template(self, title, last, end):
        self.append(title + '\n')
        self.end = '' if last else end
        return self

    def Section(self, title, last=False):
        return self._template(title, last, '\n\n')

    def Subsection(self, title, last=False):
        return self._template(title, last, '\n')

    def Bulletin(self, *sentences):
        self.append('\n'.join(sentences) + '\n')

    def Paragraph(self, *sentences):
        self.append('  '.join(sentences) + '\n')
