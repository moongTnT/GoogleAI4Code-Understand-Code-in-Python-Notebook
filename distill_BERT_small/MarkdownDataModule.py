import json
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader

from MarkdownDataset import MarkdownDataset


class MarkdownDataModule():

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

        self.data_dir = Path('..//input/')

        train_df_mark = (pd.read_csv(self.train_mark_path)
                         .drop('parent_id', axis=1)
                         .dropna()
                         .reset_index(drop=True))[:self.nrows]
        val_df_mark = (pd.read_csv(self.val_mark_path)
                       .drop('parent_id', axis=1)
                       .dropna()
                       .reset_index(drop=True))[:self.nrows]

        train_fts = json.load(open(self.train_features_path))
        val_fts = json.load(open(self.val_features_path))

        self.train_ds = MarkdownDataset(train_df_mark,
                                        model_name_or_path=self.model_name_or_path,
                                        md_max_len=self.md_max_len,
                                        total_max_len=self.total_max_len,
                                        fts=train_fts)
        self.val_ds = MarkdownDataset(val_df_mark,
                                      model_name_or_path=self.model_name_or_path,
                                      md_max_len=self.md_max_len,
                                      total_max_len=self.total_max_len,
                                      fts=val_fts)

    def get_train_loader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=False,
                          drop_last=True)

    def get_val_loader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=False,
                          drop_last=False)

    def get_df_orders(self):
        return pd.read_csv(self.data_dir / 'train_orders.csv',
                           index_col='id',
                           squeeze=True).str.split()

    def get_val_df(self):
        return pd.read_csv(self.val_path)
