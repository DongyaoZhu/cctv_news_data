from pathlib import Path
from tqdm import tqdm
from collections import defaultdict as dfd
import numpy as np
import os
import re
import csv
import math
import argparse
import sentencepiece as spm
try:
    import pkuseg
except:
    pass

EOF = '<EOF>'
SOF = '<SOF>'
UNK = '<UNK>'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', '-l', default='../data',
                        help='directory to preprocess file')

    parser.add_argument('--suffix', '-x', default='_text.csv',
                        help='file ends with what suffix')

    parser.add_argument('--model', '-m', default='subword', 
                        choices=['pkuseg', 'char', 'sentence', 'subword'],
                        help='model for segmentation, [pkuseg]')
    
    parser.add_argument('--restore', '-r', default='unigram.model',
                        help='path to subword model, e.g. unigram.model, bpe.model')

    args = parser.parse_args()

    return args


class Segment:
    def __init__(self):
        pass

    def cut(self, line):
        pass


class PKUseg(Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = pkuseg.pkuseg(model_name="news")

    def cut(self, line):
        return self.model.cut(line)


class CharSeg(Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def cut(self, line):
        return [c for c in line]


class SentenceSeg(Segment):
    def __init__(self, delimiters=['.', ','], **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.delimiters = ' *' + ' *| *'.join(delimiters).replace('.', '\.') + ' *'

    def cut(self, line):
        return re.split(self.delimiters, line)


class SubwordSeg(Segment):
    def __init__(self, restore):
        super().__init__()
        self.model = spm.SentencePieceProcessor()
        self.model.load(restore)
    
    def cut(self, line):
        return self.model.encode(line, out_type=str)


def to_segment(metadata, model, args):
    vocab = dfd(int)
    vocab[EOF] = math.inf
    vocab[SOF] = math.inf
    vocab[UNK] = math.inf
    for filename in tqdm(metadata):
        new_data = ['\t'.join(['id', 'title', 'passage']) + '\n']
        with open(filename, newline='', encoding="utf-8") as file:
            for line in csv.DictReader(file, delimiter='\t'):
                for field in ['title', 'passage']:
                    line[field] = model.cut(line[field])
                    for word in line[field]:
                        vocab[word] += 1
                    line[field] = ' '.join(line[field])

                new_data.append(
                    '\t'.join([line['id'], line['title'], line['passage']]) + '\n')

        seg_name = filename.replace(args.suffix, '_%s.csv' % args.model)
        with open(seg_name, 'w', encoding="utf-8") as seg_file:
            seg_file.writelines(new_data)

    count = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
    print('vocab:')
    print(count[:10])
    print(count[-10:])

    with open('vocab_%s.txt' % args.model, 'w', encoding="utf-8") as file:
        # file.write('\n'.join(k[0]) for k in count) + '\n')
        file.write('\n'.join(k[0] + '\t%s' % k[1] for k in count) + '\n')


def read_vocab(filename, has_count=True):
    vocab = dfd(lambda: 2)
    with open(filename, encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file.readlines())):
            # vocab[line.strip()] = i + 3
            if has_count:
                word, count = line.split('\t')
            else:
                word, count = line[:-1], 0
            # print('word, count:', word, count)
            vocab[word] = i if float(count) > 8 else 2
    vocab[EOF] = 0
    vocab[SOF] = 1
    vocab[UNK] = 2
    print('vocab:', vocab[EOF], len(vocab), vocab['你'], vocab['畚'])
    return vocab


def tokenise(sentence, vocab):
    if type(sentence) == str:
        sentence = sentence.split(' ')
    sentence = [SOF] + sentence + [EOF]
    d = np.asarray([vocab[i] if i in vocab else vocab[UNK]
                    for i in sentence], dtype=np.int32)
    return d


def to_numpy(vocab, metadata, args):
    data_title = []
    for filename in tqdm(metadata):
        seg_name = filename.replace(args.suffix, '_%s.csv' % args.model)
        with open(seg_name, newline='', encoding="utf-8") as seg_file:
            location = filename.replace(args.suffix, '_%s' % (args.model))
            os.makedirs(location, exist_ok=True)
            for line in csv.DictReader(seg_file, delimiter='\t'):

                t = tokenise(line['title'], vocab)
                t_name = '%s_t.npy' % line['id']
                np.save('/'.join([location, t_name]), t)

                p = tokenise(line['passage'], vocab)
                p_name = '%s_p.npy' % line['id']
                np.save('/'.join([location, p_name]), p)

    vocab_r = list(vocab.keys())

    def to_string(line):
        return ''.join(vocab_r[i] if i != ' ' else ' ' for i in line)
    print(filename)
    print(to_string(t))
    print(to_string(p))

def main():
    args = parse_args()

    print('using model %s' % args.model)
    if args.model == 'pkuseg':
        model = PKUseg()
    elif args.model == 'char':
        model = CharSeg()
    elif args.model == 'sentence':
        model = SentenceSeg(delimiters=['.', ',', '，', '。', '、'])
    elif args.model == 'subword':
        model = SubwordSeg(restore=args.restore)

    depth = 1
    metadata = Path(args.location).glob('/'.join('*'*depth) + args.suffix)
    metadata = list(map(str, metadata))

    total = len(metadata)
    print('data total:', total, metadata[0])

    to_segment(metadata, model, args)

    # metadata = ['../cctv_news_data/2017-11-08_text.csv']

    vocab = read_vocab('vocab_%s.txt' % args.model)

    to_numpy(vocab, metadata, args)

if __name__ == '__main__':
    main()
