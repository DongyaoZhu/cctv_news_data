from pathlib import Path
from tqdm import tqdm
from collections import defaultdict as dfd
import numpy as np
import os
import csv
import re
import math
import argparse
import pickle

EOF = '<EOF>'
SOF = '<SOF>'
UNK = '<UNK>'

SUBWORD_VOCAB_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', '-l', default='../cctv_news_data',
                        help='directory to preprocess file')

    parser.add_argument('--suffix', '-x', default='_pkuseg.csv',
                        help='file ends with what suffix, e.g. _pkuseg.csv')

    parser.add_argument('--steps', '-n', type=int, default=1000,
                        help='num merge')

    parser.add_argument('--restore', '-r', default='vocab.pkl')

    parser.add_argument('--save', '-s', default='vocab.pkl')

    args = parser.parse_args()
    return args

def get_stats(vocab):
    pairs = dfd(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def train(metadata, args, vocab=None):
    if vocab is None:
        print('loading initial vocab from *%s files' % args.suffix)
        vocab = dfd(int)
        for filename in tqdm(metadata):
            with open(filename, newline='', encoding="utf-8") as seg_file:
                for line in csv.DictReader(seg_file, delimiter='\t'):
                    t = line['title']
                    p = line['passage']
                    for x in [t, p]:
                        for y in x.split(' '):
                            vocab[' '.join(y) + ' ' + EOF] += 1
    v2 = sorted(vocab.items(), key=lambda t: t[1])
    print('initial vocab:')
    print(v2[:5])
    print(v2[-5:])

    for i in tqdm(range(args.steps)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        # v1 = vocab
        vocab = merge_vocab(best, vocab)
        if (i+1) % 50 == 0:
            print('best at iter[%d] = %s, vocab size = %d' % (i+1, best, len(vocab)))
            print(sorted(vocab.items(), key=lambda t: t[1])[-5:])
        # print(sorted(vocab.items(), key=lambda t: t[1])[-5:])
        # print('diff:', set(vocab.keys()) - set(v1.keys()))
        
    return vocab


def main():
    args = parse_args()

    depth = 1
    metadata = Path(args.location).glob('/'.join('*'*depth) + args.suffix)
    metadata = list(map(str, metadata))

    total = len(metadata)
    print('data total:', total, metadata[0])

    try:
        with open(args.restore, 'rb') as file:
            vocab = pickle.load(file)
        print('restore from [%s] successful' % args.restore)
    except Exception as e:
        vocab = None
        print(e)
        print('DID NOT restore from [%s]' % args.restore)
        
    vocab = train(metadata, args, vocab=vocab)

    try:
        with open(args.save, 'wb') as file:
            pickle.dump(vocab, file)
        print('save to [%s] successful' % args.save)
    except Exception as e:
        print(e)
        print('DID NOT save to [%s]' % args.save)


if __name__ == '__main__':
    main()
