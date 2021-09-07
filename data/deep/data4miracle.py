import pandas as pd
import os
import json

id2smiles = pd.read_csv('SMILESstrings.csv', header=None)
id2smiles = dict(zip(id2smiles.iloc[:, 0], id2smiles.iloc[:, 1]))
# os.mkdir('/home/rzy/baselines/data/miracle')
for dir in os.listdir('/home/rzy/baselines/data'):
    if dir.startswith('deep'):
        for i in json.load(open(f'/home/rzy/baselines/data/{dir}/relation2id.json', 'r')).keys():
            os.mkdir(f'/home/rzy/baselines/data/miracle/{dir}_rel{i}')
            for _set in ['train', 'valid', 'test']:
                with open(f'/home/rzy/baselines/data/miracle/{dir}_rel{i}/{_set}.csv', 'w') as curr_f:
                    curr_f.write('drugbank_id_1,drugbank_id_2,smiles_1,smiles_2,label\n')
                    for d1,rel,d2 in pd.read_csv(f'/home/rzy/baselines/data/{dir}/{_set}_pos.txt', header=None).values.tolist():
                        if int(rel) == int(i):
                            curr_f.write(f'{d1},{d2},{id2smiles[d1]},{id2smiles[d2]},1.0\n')
                with open(f'/home/rzy/baselines/data/miracle/{dir}_rel{i}/{_set}.csv', 'a+') as curr_f:
                    for d1,rel,d2 in pd.read_csv(f'/home/rzy/baselines/data/{dir}/{_set}_neg.txt', header=None).values.tolist():
                        if int(rel) == int(i):
                            curr_f.write(f'{d1},{d2},{id2smiles[d1]},{id2smiles[d2]},0.0\n')


