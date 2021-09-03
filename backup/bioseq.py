import pandas as pd

bio_ids = list(set(pd.read_csv('druglist.csv', header=None).iloc[:, 0])\
    .difference(set(pd.read_csv('SMILESstrings.csv', header=None).iloc[:, 0])))
bio_seqs = pd.read_csv('all_DBxx_seq.csv', header=None)
bio_seqs = dict(zip(bio_seqs.iloc[:, 0], bio_seqs.iloc[:,1]))
bio2seq = []
for id in bio_ids:
    if id in bio_seqs:
        bio2seq.append([id, bio_seqs[id]])
    else:
        print(id)
pd.DataFrame(bio2seq).to_csv('biotech_seqs.csv', index=False, header=None)
