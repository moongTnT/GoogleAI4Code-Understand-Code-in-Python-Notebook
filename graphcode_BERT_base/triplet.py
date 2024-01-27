import numpy as np
from tqdm import tqdm


def generate_triplets(df, args, mode='train'):
    print(f'generate {mode} triplets')

    triplets = []
    drop_sz = 1000 if args.debug else 10000
    random_drop = np.random.random(size=drop_sz) > .9
    count = 0

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']

        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values

        for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
            # cell_id의 마크다운 바로 뒤에 나오는 코드셀이면 True
            labels = np.array([(r == (rank+1)) for r in df_tmp_code_rank]).astype(int)

            for cid, label in zip(df_tmp_code_cell_id, labels):
                count += 1
                if label == 1 or random_drop[count % drop_sz] or mode=='test':
                    triplets.append([cell_id, cid, label])
    
    return triplets

