import json
import os
from pathlib import Path
import re
import joblib
import numpy as np

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

import nltk
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, **args):
        self.__dict__.update(args)
        self.data_dir = Path(self.input_path)

    def read_notebook(self, path):
        return (
            pd.read_json(path, dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
        )

    def get_ranks(self, base, derived):
        return [base.index(d) for d in derived]

    def run(self, mode='train', nvalid=0.1):
        if os.path.exists(self.train_path) and os.path.exists(self.val_path):
            print('train_df, val_df are already exits')
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path) 
            train_df_mark = pd.read_csv(self.train_mark_path)
            val_df_mark = pd.read_csv(self.val_mark_path)
            return train_df, val_df, train_df_mark, val_df_mark

        paths = list((self.data_dir / mode).glob('*.json'))
        notebooks = [self.read_notebook(path)
                     for path in tqdm(paths, desc=f'{mode} NBs')]

        df = (pd.concat(notebooks)
              .set_index('id', append=True)
              .swaplevel()
              .sort_index(level='id', sort_remaining=False))

        df_orders = pd.read_csv(
            self.data_dir / 'train_orders.csv',
            index_col='id',
            squeeze=True).str.split()

        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right'
        )

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id,
                          'rank': self.get_ranks(cell_order, cell_id)}

        df_ranks = (
            pd.DataFrame
            .from_dict(ranks, orient='index')
            .rename_axis('id')
            .apply(pd.Series.explode)
            .set_index('cell_id', append=True)
        )

        df_ancestors = pd.read_csv(
            self.data_dir / 'train_ancestors.csv', index_col='id')
        df = df.reset_index().merge(
            df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])
        df['pct_rank'] = df['rank'] / \
            df.groupby('id')['cell_id'].transform('count')

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=nvalid, random_state=0)

        train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))
            
        train_df = df.loc[train_ind].reset_index(drop=True)
        val_df = df.loc[val_ind].reset_index(drop=True)

        train_df_mark = train_df[train_df['cell_type'] == 'markdown'].reset_index(drop=True)
        val_df_mark = val_df[val_df['cell_type'] == 'markdown'].reset_index(drop=True)

        train_df_mark.to_csv(self.train_mark_path + f'_fold{i}')
        val_df_mark.to_csv(self.val_mark_path + f'_fold{i}')

        train_df.to_csv(self.train_path + f'_fold{i}')
        val_df.to_csv(self.val_path + f'_fold{i}')

        return train_df, val_df, train_df_mark, val_df_mark


class _20CodeCellPreprocessor(Preprocessor):
    def __init__(self, **args):
        self.__dict__.update(args)
        super(_20CodeCellPreprocessor, self).__init__(**args)
        
    def clean_code(self, cell):
        return str(cell).replace('\\n', '\n')

    def sample_cells(self, cells, n=20):
        cells = [self.clean_code(cell) for cell in cells]

        if n >= len(cells): # 코드 셀이 20개 이하라면 그냥 반환
            return [cell[:200] for cell in cells]
        else:
            results = []
            step = len(cells) / n # 총 20개의 코드셀이 샘플링 되도록 스텝을 조절
            idx = 0
            while int(np.round(idx) < len(cells)):
                results.append(cells[int(np.round(idx))])
                idx += step
            assert cells[0] in results # 첫번쨰 코드셀은 반드시 들어가야 한다?
            if cells[-1] not in results: # 말전 코드셀은 반드시 들어가야 한다?
                results[-1] = cells[-1]
            return results

    def get_features(self, df):
        features = dict()
        df = df.sort_values('rank').reset_index(drop=True)

        for idx, sub_df in tqdm(df.groupby('id')):
            features[idx] = dict()
            total_md = sub_df[sub_df.cell_type == 'markdown'].shape[0]
            code_sub_df = sub_df[sub_df.cell_type == 'code']
            total_code = code_sub_df.shape[0]
            codes = self.sample_cells(code_sub_df.source.values, 20)
            features[idx]['total_code'] = total_code
            features[idx]['total_md'] = total_md
            features[idx]['codes'] = codes

        # features = {
        #     노트북id: {
        #         'total_code': 코드 셀의 개수,
        #         'total_md': 마크다운 샐의 개수,
        #         'codes': [코드셀0, 코드셀1, ... , 코드셀 19]
        #     },
        #     ...
        # }
        return features

    def run(self):
        train_df, val_df, train_df_mark, val_df_mark = super().run()

        if os.path.exists(self.train_features_path) and os.path.exists(self.val_features_path):
            print('train_fts, val_fts are already exists')
            train_fts = json.load(open(self.train_features_path))
            val_fts = json.load(open(self.val_features_path))
        else:
            train_fts = self.get_features(train_df)
            val_fts = self.get_features(val_df)          
            json.dump(train_fts, open(self.train_features_path,"wt"))
            json.dump(val_fts, open(self.val_features_path,"wt"))    

        return train_df, val_df, train_df_mark, val_df_mark, train_fts, val_fts


class PairwisePreprocessor(Preprocessor):
    def __init__(self, **args):
        self.__dict__.update(args)
        super(PairwisePreprocessor, self).__init__(**args)

        nltk.download('wordnet')
        nltk.download('omw-1.4')

        self.stemmer = WordNetLemmatizer()

    def preprocess_text(self, document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        # return document

        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def run(self):
        if os.path.exists(self.pairwise_train_path) and os.path.exists(self.pairwise_val_path):
            print('pairwise_train_df and val_df already exists')
            train_df = pd.read_csv(self.pairwise_train_path)
            val_df = pd.read_csv(self.pairwise_val_path)
        else:
            print('generate_train_df and val_df')
            train_df, val_df, _, _ = super().run()
        
            train_df.source = train_df.source.apply(self.preprocess_text)
            val_df.source = val_df.source.apply(self.preprocess_text)

            train_df.to_csv(self.pairwise_train_path)
            val_df.to_csv(self.pairwise_val_path)

        if os.path.exists(self.dict_cellid_source_path):
            dict_cellid_source = joblib.load(self.dict_cellid_source_path)
        else:
            df = pd.concat([train_df, val_df])
            dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))
            joblib.dump(dict_cellid_source, self.dict_cellid_source_path)
        
        return train_df, val_df, dict_cellid_source

