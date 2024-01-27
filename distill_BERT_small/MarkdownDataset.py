import joblib
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):
    def __init__(self, df, max_len, num_train, mode='train'):
        super().__init__()
        
        if num_train == None:
            self.path = './data-all'
        else:
            self.path = f'./data-{num_train}'
        
        self.df = df
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small", do_lower_case=True)
        self.mode = mode
        self.dict_cellid_source = joblib.load(self.path + '/dict_cellid_source.pkl')

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        txt = self.dict_cellid_source[row[0]] + '[SEP]' + self.dict_cellid_source[row[1]]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)