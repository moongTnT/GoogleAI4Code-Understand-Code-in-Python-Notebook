import os
import sys
import numpy as np
from awp import AWP
from earlystopping import EarlyStopping
from scheduler import CosineAnnealingWarmupRestarts
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from metrics import kendall_tau
from model import PairwiseModel, _20SampleModel


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout, position=0, leave=True)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, optimizer, scheduler, scaler, val_df, df_orders, args):
    criterion = torch.nn.L1Loss()

    for e in range(args.epoch, 100):
        model.train()

        tbar = tqdm(train_loader, file=sys.stdout, position=0, leave=True)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()

            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f'Epoch {e+1} Loss: {avg_loss} lr: {scheduler.get_lr()}')

        y_val, y_pred = validate(model, val_loader)
        val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
        val_df.loc[val_df['cell_type'] == 'markdown', 'pred'] = y_pred
        y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
        preds_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", preds_score)

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        torch.save(model.state_dict(), args.output_path + f'/model_epoch_{e}_{preds_score}.bin')


def get_preds(preds, val_df):
    pred_vals = []
    count = 0
    for id, df_tmp in tqdm(val_df.groupby('id')):
        df_tmp_mark = df_tmp[df_tmp['cell_type'] == 'markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
        df_tmp_code_rank = df_tmp_code['rank'].rank().values
        N_code = len(df_tmp_code_rank)
        N_mark = len(df_tmp_mark)

        preds_tmp = preds[count:count + N_mark*N_code]
        count += N_mark*N_code

        for i in range(N_mark):
            pred = preds_tmp[i*N_code:i*N_code + N_code]

            softmax = np.exp((pred-np.mean(pred)) * 20) / \
                np.sum(np.exp((pred-np.mean(pred)) * 20))

            rank = np.sum(softmax * df_tmp_code_rank)
            pred_vals.append(rank)
    return pred_vals


def pairwise_validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout, position=0, leave=True)

    preds = np.zeros(len(val_loader.dataset), dtype='float32')
    print('preds.shape:', preds.shape)
    labels = []
    count = 0

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            # with torch.cuda.amp.autocast():
            #     pred = model(*inputs)

            pred = model(*inputs)
            pred = pred.detach().cpu().numpy().ravel()

            # for debugging
            print(f'{idx} pred_shape:', pred.shape)
            if idx == 0:
                print(pred)

            preds[count:count+len(pred)] = pred
            print(preds)
            count += len(pred)

            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), preds


def pairwise_train(model, train_loader, val_loader, optimizer, scheduler, scaler, val_df, df_orders, args):
    criterion = torch.nn.BCELoss()

    for e in range(args.epoch, args.epochs):
        model.train()

        tbar = tqdm(train_loader, file=sys.stdout, position=0, leave=True)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            # with torch.cuda.amp.autocast():
            #     pred = model(*inputs)
            #     loss = criterion(pred, target)
            # scaler.scale(loss).backward()
            optimizer.zero_grad()

            pred = model(*inputs)
            loss = criterion(pred, target)
            loss.backward()

            # scaler.step(optimizer)
            # scaler.update()

            optimizer.step()
            scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f"Epoch {e+1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        y_val, y_pred = pairwise_validate(model, val_loader)
        y_pred = get_preds(y_pred, val_df)
        val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
        val_df.loc[val_df['cell_type'] == 'markdown', 'pred'] = y_pred
        y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
        
        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, args.output_path + f'/chekcpoint_{e}.pt')
        torch.save(model.state_dict(), args.output_path +
                   f'/model_epoch_{e}.bin')

    return model


def train_setup(args):
    model = _20SampleModel(model_path=args.model_name_or_path)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    num_train_optimization_steps = args.num_train_steps
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=3e-5, correct_bias=False)
    
    scheduler = (
        CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=num_train_optimization_steps,
            cycle_mult=1,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=num_train_optimization_steps * 0.2,
            gamma=1.,
            last_epoch=-1
        ))  # Pytorch scheduler

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0.05*num_train_optimization_steps,
    #     num_training_steps=num_train_optimization_steps,
    # )
    
    scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, scheduler, scaler


def pairwise_train_setup(args):
    model = PairwiseModel(model_path=args.model_name_or_path)
    num_train_optimization_steps = args.num_train_steps
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=3e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8)  # 1e-08)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.05*num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, scheduler, scaler
