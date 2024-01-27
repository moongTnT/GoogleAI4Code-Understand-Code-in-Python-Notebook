import argparse
import gc

import pandas as pd
import torch
from dataset import pairwise_data_setup

from preprocessor import PairwisePreprocessor, _20CodeCellPreprocessor
from train import pairwise_train, pairwise_train_setup
from util import pairwise_debug_setup


def main():
    parser = argparse.ArgumentParser(description='Process some arguments')

    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')

    parser.add_argument('--input_path', type=str, default='../input/')

    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
    parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')

    parser.add_argument('--val_path', type=str, default="./data/val.csv")
    parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
    parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')

    parser.add_argument('--pairwise_train_path', type=str, default='./data/pairwise_train.csv')
    parser.add_argument('--pairwise_val_path', type=str, default="./data/pairwise_val.csv")
    parser.add_argument('--dict_cellid_source_path', type=str, default="./data/dict_cellid_source.pkl")
    
    parser.add_argument('--output_path', type=str, default='./output-pairwise-codebert')

    parser.add_argument('--md_max_len', type=int, default=64)
    parser.add_argument('--total_max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--load_train', type=bool, default=False)

    args = parser.parse_args()

    preprocessor = PairwisePreprocessor(**vars(args))
    train_df, val_df, _ = preprocessor.run()

    print('before debug', train_df.shape, val_df.shape)

    if args.debug:
        train_df, val_df = pairwise_debug_setup(train_df, val_df)

    print('after debug', train_df.shape, val_df.shape)

    train_loader, val_loader = pairwise_data_setup(train_df, val_df, args)

    df_orders = pd.read_csv(args.input_path + 'train_orders.csv',
                            index_col='id',
                            squeeze=True).str.split()

    del train_df, _, preprocessor
    gc.collect()

    args.num_train_steps = args.epochs * len(train_loader)

    model, optimizer, scheduler, scaler = pairwise_train_setup(args)

    if args.load_train:
        checkpoint = torch.load(args.checkpoint_path)
        args.epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

        model.cuda()

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        model.cuda()

    pairwise_train(model, train_loader, val_loader, optimizer, scheduler, scaler, val_df, df_orders, args)

if __name__ == '__main__':
    main()
