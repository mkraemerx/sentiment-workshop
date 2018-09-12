import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import math
import numpy as np
import pandas as pd
import spacy
from torchtext import data


class TwitterHandleMatcher(object):
    def __init__(self):
        self.pos_value = 95  # PROPN

    def __call__(self, doc):
        for i, t in enumerate(doc):
            if t.text.startswith('@'):
                span = doc[i:i+1]
                span.merge(pos=self.pos_value)
        return doc


class TwitterPipeline:

    TEXT = 'text'
    LABEL = 'label'

    def __init__(self, lang_model='de'):

        self.nlp = spacy.load(lang_model)
        twitter_matcher = TwitterHandleMatcher()
        self.nlp.add_pipe(twitter_matcher, after='tagger')

    def process_data(self, filename_in, fields, split_ratio=None):
        """
        processes one input data file
        if split_ratio is given, the file is split into 2 separate lists

        - filename_in the name of the file to process
        - list of field tuples for creation of Examples
        - returns dict of torchtext.data.Example
        """

        def str2bool_func(s): return s == 'OFFENSE'

        # read and format data as necessary
        df_raw = pd.read_csv(filename_in, sep='\t', header=None, names=[
            self.TEXT, self.LABEL, 'detail']).drop('detail', axis=1)
        df_raw[self.LABEL] = df_raw[self.LABEL].apply(str2bool_func)

        # split if needed
        if (split_ratio):
            df_list = self.split_off_validation_data(df_raw, split_ratio)
        else:
            df_list = [df_raw]

        result_list = list()  # list of example lists (e.g. for train and validation)
        for df in df_list:

            ex_list = list()

            for _, row in df.iterrows():
                doc = self.nlp(row[self.TEXT])
                text = [str.lower(t.text) for t in doc]
                pos = [t.pos for t in doc]
                lemma = [t.lemma_ for t in doc]
                label = row[self.LABEL]
                ex_list.append(data.Example.fromlist(
                    [text, pos, lemma, label], fields))
            result_list.append(ex_list)
        return tuple(result_list)

    def split_off_validation_data(self, df_in, ratio):
        np.random.seed(762)
        idx = np.arange(len(df_in))
        np.random.shuffle(idx)
        len_val = math.ceil(len(df_in) * ratio)

        df_val = df_in.iloc[idx[:len_val]]
        df_trn = df_in.iloc[idx[len_val:]]

        return [df_trn, df_val]
