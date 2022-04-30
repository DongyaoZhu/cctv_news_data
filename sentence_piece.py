from pathlib import Path
from tqdm import tqdm
from collections import defaultdict as dfd
import numpy as np
import os
import io
import csv
import math
import argparse
import pickle
import sentencepiece as spm


EOF = '<EOF>'
SOF = '<SOF>'
UNK = '<UNK>'

SUBWORD_VOCAB_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', '-l', default='../data',
                        help='directory to preprocess file')

    parser.add_argument('--suffix', '-x', default='_text.csv',
                        help='file ends with what suffix, e.g. _pkuseg.csv')

    parser.add_argument('--model', '-m', default='unigram',
                        choices=['unigram', 'bpe', 'word'],
                        help='model for sentencepiece')

    parser.add_argument('--steps', '-n', type=int, default=1000,
                        help='num merge')

    parser.add_argument('--restore', '-r', default='vocab.pkl')

    args = parser.parse_args()
    return args


def get_data(path, suffix):

    depth = 1
    metadata = Path(path).glob('/'.join('*'*depth) + suffix)
    metadata = list(map(str, metadata))

    total = len(metadata)
    print('data total:', total, metadata[0])

    def gen():
        while True:
            for filename in (metadata[:]):
                with open(filename, newline='', encoding="utf-8") as seg_file:
                    for line in csv.DictReader(seg_file, delimiter='\t'):
                        for field in ['title', 'passage']:
                            yield line[field]
            break
    return metadata, gen()

def train(data, args):
    # data = iter(data)
    print('d:', next(data))

    model = io.BytesIO()
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(args.restore)
        print('[INFO] restore model at %s' % args.restore)
    except:
        print('[INFO] new model training')
        spm.SentencePieceTrainer.train(
            sentence_iterator=data, 
            model_writer=model, 
            vocab_size=32000,
            model_type=args.model,
            train_extremely_large_corpus=True,
            max_sentence_length=12800)

        sp = spm.SentencePieceProcessor(model_proto=model.getvalue())

        # Serialize the model as file.
        with open('sp_%s.model' % args.model, 'wb') as f:
            f.write(model.getvalue())

    vocabs = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(len(vocabs))
    with open('vocab_sp_%s.txt' % args.model, 'w', encoding="utf-8") as file:
        file.write('\n'.join(vocabs))

    test_sentences = [
        '国际联播快讯',
        '当温家宝乘坐的专机抵达苏黎世机场时，瑞士联邦副主席兼经济部长洛伊特哈德女士等在舷梯旁迎候。中国驻瑞士大使董津义、驻日内瓦代表团团长李保东、驻世界贸易组织代表孙振宇，以及中国驻瑞士使领馆工作人员、留学生代表也前来迎接。',
    ]
    for s in test_sentences:
        s1 = '#'.join(map(str, sp.encode(s)))
        s2 = '#'.join(map(lambda i: vocabs[i], sp.encode(s)))
        s3 = '#'.join(sp.encode(s, out_type=str))
        print(' '.join([s1, s2, s3]))

    return sp


def to_numpy(sp, metadata, args):
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print('initial vocab:', len(vocab))
    print(vocab[:5])
    print(vocab[-5:])

    data_title = []
    for filename in tqdm(metadata):
        with open(filename, newline='', encoding="utf-8") as seg_file:
            location = filename.replace(args.suffix, '_%s' % (args.model))
            os.makedirs(location, exist_ok=True)
            for line in csv.DictReader(seg_file, delimiter='\t'):

                t = sp.encode(line['title'])
                t_name = '%s_t.npy' % line['id']
                np.save('/'.join([location, t_name]), t)

                p = sp.encode(line['passage'])
                p_name = '%s_p.npy' % line['id']
                np.save('/'.join([location, p_name]), p)

    def to_string(line):
        return '#'.join(map(lambda i: vocab[i], line))
    print(filename)
    print(to_string(t))
    print(to_string(p))

    return vocab


def main():
    args = parse_args()

    metadata, data = get_data(args.location, args.suffix)

    sp = train(data, args)

    to_numpy(sp, metadata, args)

if __name__ == '__main__':
    main()
