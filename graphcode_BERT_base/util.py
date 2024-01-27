import pandas as pd

def _20sample_debug_setup(train_df, train_df_mark, train_fts, val_df, val_df_mark, val_fts):    
    train_sample_id = train_df['id'].drop_duplicates().sample(frac=0.1).values
    train_df = train_df[train_df['id'].isin(train_sample_id)].reset_index(drop=True)
    train_df_mark = train_df_mark[train_df_mark['id'].isin(train_sample_id)].reset_index(drop=True)
    train_fts = {k: v for k, v in train_fts.items()if k in train_sample_id}

    val_sample_id = val_df['id'].drop_duplicates().sample(frac=0.1).values
    val_df = val_df[val_df['id'].isin(val_sample_id)].reset_index(drop=True)
    val_df_mark = val_df_mark[val_df_mark['id'].isin(val_sample_id)].reset_index(drop=True)
    val_fts = {k: v for k, v in val_fts.items() if k in val_sample_id}

    return train_df, train_df_mark, train_fts, val_df, val_df_mark, val_fts

def pairwise_debug_setup(train_df, val_df):
    train_sample_id = train_df['id'].drop_duplicates().sample(frac=0.1).values
    train_df = train_df[train_df['id'].isin(train_sample_id)].reset_index(drop=True)
    
    val_sample_id = val_df['id'].drop_duplicates().sample(frac=0.1).values
    val_df = val_df[val_df['id'].isin(val_sample_id)].reset_index(drop=True)
    
    return train_df, val_df
