import os
import math
import numpy as np
import pandas as pd
import argparse
from collections import Counter

import spacy

# inspired by https://github.com/mesnilgr/nbsvm

nlp = spacy.load('de')


def tokenize(sentence, grams):
    words = [t.text for t in nlp(sentence)]
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens


def build_counts(doc_list, grams):
    counts = Counter()
    for sentence in doc_list:
        counts.update(tokenize(sentence, grams))
    return counts


def process_files(df, dic, r, outfn, grams):
    # create the appropriate files for liblinear
    output = []
    for num, line_in in df.iterrows():
        tokens = tokenize(line_in['text'], grams)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexes.sort()
        if line_in['bin']:
            line_out = ['1']
        else:
            line_out = ['-1']
        for i in indexes:
            line_out += ["%i:%f" % (i + 1, r[i])]
        output += [" ".join(line_out)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(list(poscounts.keys()) + list(negcounts.keys())))
    print(f'vocab len {len(alltokens)}')
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print('computing r..')
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r


def main(ftrain, ftest, out, liblinear, ngram):
    validation_ratio = 0.1
    ngram = [int(i) for i in ngram]
    print('counting (train only)..')
    df_trn = pd.read_csv(ftrain, sep='\t', header=None, names=[
                         'text', 'bin', 'detail']).drop('detail', axis=1)
    df_trn['bin'] = df_trn['bin'] == 'OFFENSE'

    idx = np.arange(len(df_trn))
    np.random.shuffle(idx)
    val_size = math.ceil(len(df_trn) * validation_ratio)

    df_test = df_trn.iloc[idx[:val_size]]
    df_trn = df_trn.iloc[idx[val_size:]]

    ptrain = df_trn[df_trn['bin']]
    ntrain = df_trn[df_trn['bin'] == False]
    poscounts = build_counts(ptrain['text'], ngram)
    negcounts = build_counts(ntrain['text'], ngram)
    dic, r = compute_ratio(poscounts, negcounts)

    print('processing files..')
    process_files(df_trn, dic, r, 'train-nbsvm.txt', ngram)
    process_files(df_test, dic, r, 'test-nbsvm.txt', ngram)

    trainsvm = os.path.join(liblinear, "train")
    predictsvm = os.path.join(liblinear, "predict")

    os.system(trainsvm + " -s 2 train-nbsvm.txt model.logreg")
    os.system(predictsvm + " test-nbsvm.txt model.logreg " + out)
    #os.system("rm model.logreg train-nbsvm.txt test-nbsvm.txt")


if __name__ == "__main__":
    """
    Usage :
    python nbsvm.py --liblinear /PATH/liblinear-1.96\
        --ftrain /PATH/data/full-train.tsv\
        --ftest /PATH/data/test.tsv\
         --ngram 123 --out TEST-SCORE
    """

    parser = argparse.ArgumentParser(
        description='Run NB-SVM on some text files.')
    parser.add_argument(
        '--liblinear', help='path of liblinear install e.g. */liblinear-1.96')
    parser.add_argument(
        '--ftrain', help='path of the text file TRAIN')
    parser.add_argument(
        '--ftest', help='path of the text file TEST')
    parser.add_argument('--out', help='path and fileename for score output')
    parser.add_argument(
        '--ngram', help='N-grams considered e.g. 123 is uni+bi+tri-grams')
    args = vars(parser.parse_args())

    main(**args)
