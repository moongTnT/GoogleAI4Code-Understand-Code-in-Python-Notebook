import joblib
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

from triplet import generate_triplets


class _20SampleDataset(Dataset):

    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len
        self.fts = fts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]['codes']],
            add_special_tokens=True,
            max_length=23,
            padding='max_length',
            truncation=True
        )

        n_md = self.fts[row.id]['total_md']
        n_code = self.fts[row.id]['total_code']
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == len(mask)

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


class PairwiseDataset(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.max_len = args.total_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.dict_cellid_source = joblib.load(args.dict_cellid_source_path)

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        txt = self.dict_cellid_source[row[0]] + \
            '[SEP]' + self.dict_cellid_source[row[1]]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)


def _20sample_data_setup(train_df_mark, val_df_mark, train_fts, val_fts, args):
    train_ds = _20SampleDataset(train_df_mark,
                                model_name_or_path=args.model_name_or_path,
                                md_max_len=args.md_max_len,
                                total_max_len=args.total_max_len,
                                fts=train_fts)
    val_ds = _20SampleDataset(val_df_mark,
                              model_name_or_path=args.model_name_or_path,
                              md_max_len=args.md_max_len,
                              total_max_len=args.total_max_len,
                              fts=val_fts)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers,
                              pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            pin_memory=False,
                            drop_last=False)
    return train_loader, val_loader


def pairwise_data_setup(train_df, val_df, args):
    train_triplets = generate_triplets(train_df, args, mode='train')
    # test 모드는 drop없이 다 때려박기 때문에 데이터 개수가 많다.
    val_triplets = generate_triplets(val_df, args, mode='test') 

    train_ds = PairwiseDataset(train_triplets, args)
    val_ds = PairwiseDataset(val_triplets, args)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False,
                              drop_last=True)

    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    return train_loader, val_loader
