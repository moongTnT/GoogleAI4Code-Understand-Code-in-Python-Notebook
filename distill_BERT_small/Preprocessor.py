import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

class Preprocessor:
    
    def __init__(self, nrows):
        self.data_dir = Path('..//input/')  
        if not os.path.exists('./data'):
            os.mkdir('./data')
        self.nrows = nrows

    def read_notebook(self, path):
        return (
            pd.read_json(
                path,
                dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
        )

    def get_notebooks_train(self):
        paths_train = list((self.data_dir / 'train').glob('*.json'))[:self.nrows]
        notebooks_train = [
            self.read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
        ]
        return notebooks_train

    def get_train_df(self):
        notebooks_train = self.get_notebooks_train()
        df = (
            pd.concat(notebooks_train)
                .set_index('id', append=True)
                .swaplevel()
                .sort_index(level='id', sort_remaining=False)
        )
        return df

    def get_ranks(self, base, derived):
        return [base.index(d) for d in derived]

    def get_df_orders(self):
        df_orders = pd.read_csv(
            self.data_dir / 'train_orders.csv',
            index_col='id', 
            squeeze=True
        ).str.split() # Split the string representation of cell_ids into a list
        return df_orders

    def get_ranks_dict(self, df):
        df_orders = self.get_df_orders()
        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right'
        )
        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id, 'rank': self.get_ranks(cell_order, cell_id)}

        return ranks

    def get_df_ranks(self, df):
        ranks = self.get_ranks_dict(df)
        df_ranks = (
            pd.DataFrame
                .from_dict(ranks, orient='index')
                .rename_axis('id')
                .apply(pd.Series.explode)
                .set_index('cell_id', append=True)
        )
        return df_ranks

    def get_final_df(self):
        df = self.get_train_df()
        df_ranks = self.get_df_ranks(df)

        df_ancestors = pd.read_csv(self.data_dir / 'train_ancestors.csv', index_col='id')
        df = df.reset_index().merge(df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])
        df['pct_rank'] = df['rank'] / df.groupby('id')['cell_id'].transform('count')

        return df

    def split_and_save(self, nvalid=0.1):
        df = self.get_final_df()

        splitter = GroupShuffleSplit(n_splits=1, test_size=nvalid, random_state=0)
        
        train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))
        
        train_df = df.loc[train_ind].reset_index(drop=True)
        val_df = df.loc[val_ind].reset_index(drop=True)

        # Base markdown dataframes
        train_df_mark = train_df[train_df['cell_type'] == 'markdown'].reset_index(drop=True)
        val_df_mark = val_df[val_df['cell_type'] == 'markdown'].reset_index(drop=True)
        
        train_df_mark.to_csv('./data/train_mark.csv')
        val_df_mark.to_csv('./data/val_mark.csv')
        
        train_df.to_csv('./data/train.csv')
        val_df.to_csv('./data/val.csv')

        print(train_df.head())

        return train_df, val_df
        
    def run(self):
        self.split_and_save()
