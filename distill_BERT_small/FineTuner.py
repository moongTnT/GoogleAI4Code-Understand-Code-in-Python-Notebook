import sys
from turtle import position
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from MarkdownDataset import MarkdownDataset


class FineTuner():
    def __init__(self, df, epochs):
        self.df = df
        self.epochs = epochs

    def generate_triplet(self, mode='train'):
        """
        학습 혹은 테스트용 데이터 triplet 생성 함수
        mode: 'train' | 'test'
        result: triplet = [markdown_id, codecell_id, is_next]의 리스트
        """
        triplets = []
        ids = self.df.id.unique()
        random_drop = np.random.random(size=10000) > .9
        count = 0

        for id, df_tmp in tqdm(self.df.groupby('id'), position=0, leave=True):
            df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']

            df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
            df_tmp_code_rank = df_tmp_code['rank'].values
            df_tmp_code_cell_id = df_tmp_code['cell_id'].values

            for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
                labels = np.array([(r == (rank+1))
                                  for r in df_tmp_code_rank]).astype('int')

                for cid, label in zip(df_tmp_code_cell_id, labels):
                    count += 1
                    if label == 1:
                        triplets.append([cell_id, cid, label])
                    elif mode == 'test':
                        triplets.append([cell_id, cid, label])
                    elif random_drop[count % 10000]:
                        triplets.append([cell_id, cid, label])

        return triplets

    def adjust_lr(self, optimizer, epoch):
        if epoch < 1:
            lr = 5e-5
        elif epoch < 2:
            lr = 5e-5
        elif epoch < 5:
            lr = 5e-5
        else:
            lr = 5e-5

        for p in optimizer.param_groups:
            p['lr'] = lr

        return lr

    def get_optimizer(self, net):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=3e-4,
                                     betas=(0.9, 0.999),
                                     eps=1e-8)  # 1e-08)
        return optimizer

    def read_data(self, data):
        return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

    def validate(self, model, val_loader, mode='train'):
        model.eval()

        tbar = tqdm(val_loader, file=sys.stdout)

        preds = []
        labels = []

        with torch.no_grad():
            for idx, data in enumerate(tbar):
                inputs, target = self.read_data(data)

                pred = model(inputs[0], inputs[1])

                preds.append(pred.detach().cpu().numpy().ravel())
                if mode == 'test':
                    labels.append(target.detach().cpu().numpy().ravel())
        if mode == 'test':
            return np.concatenate(preds)
        else:
            return np.concatenate(labels), np.concatenate(preds)


    def train(self, model, train_loader):
        np.random.seed(0)

        optimizer = self.get_optimizer(model)
        criterion = torch.nn.L1Loss()
        scaler = torch.cuda.amp.GradScaler()

        for e in range(self.epochs):
            model.train()
            tbar = tqdm(train_loader, file=sys.stdout)

            lr = self.adjust_lr(optimizer, e)

            loss_list = []
            preds = []
            labels = []

            for idx, data in enumerate(tbar):
                inputs, target = self.read_data(data)

                if idx == 0 and e == 0:
                    print()
                    print('==== THIS IS INPUT ====')
                    print(inputs)

                    print()
                    print('==== THIS IS TARGET ====')
                    print(target)
                    print()

                with torch.cuda.amp.autocast():
                    pred = model(*inputs)
                    loss = criterion(pred, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                loss_list.append(loss.detach().cpu().item())
                preds.append(pred.detach().cpu().numpy().ravel())
                labels.append(target.detach().cpu().numpy().ravel())

                avg_loss = np.round(np.mean(loss_list), 4)

                tbar.set_description(f"Epoch {e+1} Loss: {avg_loss} lr: {lr}")
        
            output_model_file = f"./outputs/my_own_model_{e}.bin"
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)

        return model

        