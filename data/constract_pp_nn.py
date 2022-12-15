import random

import pandas as pd
import sys


def gen_pairs(dataset_name, K):
    classes = pd.unique(df['cluster_id'])
    # print(classes)
    pos_pairs = []
    neg_pairs = []
    for ci in classes:
        index_i = list(df[df['cluster_id'] == ci].index)
        print(index_i)
        random_is = random.sample(range(0, len(index_i)), min([K, len(index_i)]))
        div = (len(random_is) - len(random_is) % 2) // 2
        j_pos = random_is[:div]
        random_is = random_is[div:]
        pos_pairs += [[index_i[i], index_i[j]] for i, j in zip(random_is, j_pos)]
        j_neg = list(df[df['cluster_id'] != ci].index)
        random_j_neg = random.sample(range(0, len(j_neg)), min([K, div]))
        neg_pairs += [[index_i[i], j_neg[j]] for i, j in zip(random_is, random_j_neg)]
    print(pos_pairs)
    print(neg_pairs)
    pd.DataFrame(pos_pairs).to_csv(f'{dataset_name}/in_pairs_prot.csv', header=False, index=False)
    pd.DataFrame(neg_pairs).to_csv(f'{dataset_name}/bt_pairs_prot.csv', header=False, index=False)


dataset = sys.argv[1]
df = pd.read_csv(f'{dataset}/cluster_res-ward_n=20.csv', index_col=0, header=None, names=['cluster_id'])
print(df)
for univ in pd.unique(df['cluster_id']):
    print(df[df['cluster_id'] == univ].shape[0])

print('\n')

gen_pairs(dataset, 100)
