import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from transformers import AutoTokenizer
from Preprocessor import Preprocessor

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

class AdditionalPreprocessor(Preprocessor):

    def __init__(self, nrows):
        super().__init__(nrows)
        self.stemmer = WordNetLemmatizer()

    def get_preprocessed_df(self, path):
        return pd.read_csv(path, nrows=self.nrows)

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
        #return document

        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def get_additional_preprocessed_df(self, df):
        return df.source.apply(self.preprocess_text)
        
    def split_and_save(self, nvalid=0.1):
        df = self.get_final_df()
        df.source = self.get_additional_preprocessed_df(df)

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

    def get_textfile(self, filename, df):
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                for id, item in tqdm(df.groupby('id'), position=0, leave=True):
                    df_markdown = item[item['cell_type'] == 'markdown']
                    for source, rank in df_markdown[['source', 'rank']].values:
                        cell_source = df_markdown[df_markdown['rank'] == (rank+1)]
                        if len(cell_source):
                            sentence = source + ' [SEP] ' + cell_source.source.values[0]
                            f.write(sentence + '\n')

    def run(self):
        df = self.get_final_df()
        df.source = self.get_additional_preprocessed_df(df)
        df.to_csv('./data/train_df_all.csv')
        self.get_textfile('./data/text.txt', df)

