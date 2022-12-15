import ssl

from sklearn.cluster import AgglomerativeClustering, KMeans

ssl._create_default_https_context = ssl._create_unverified_context
import os
import pandas as pd
from tqdm import tqdm
import sklearn.cluster as sc

from rdkit.Chem import AllChem


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))


def gen_fp(sm_path, fp_path):
    df_smi = pd.read_csv(sm_path, header=None, names=['DBxx', 'smiles'])
    smiles = df_smi["smiles"]

    mgf_feat_list = []
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])
        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)

    mgf_feat = pd.DataFrame(mgf_feat_list, dtype="int64")
    mgf_feat.index = df_smi["DBxx"]
    print("morgan feature shape: ", mgf_feat.shape)
    mgf_feat.to_csv(fp_path, header=False)


def cluster_mols(ds, Xn, method, n_cluster=10):
    def rec_cluster_res(note=""):
        with open(os.path.join(ds,'cluster_res' + note + '.csv'), 'w') as f:
            for i in range(len(pred_y)):
                f.write(f'{Xn.index.values[i]},{pred_y[i]}\n')

    if method == 'MeanShift':
        bw = sc.estimate_bandwidth(Xn, n_samples=len(Xn), quantile=0.01)
        model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
        pred_y = model.fit_predict(Xn)
        rec_cluster_res()
    elif method == 'Hier':
        for linkage in ('ward', 'average', 'complete'):
            model = AgglomerativeClustering(linkage=linkage, n_clusters=n_cluster)
            pred_y = model.fit_predict(Xn)
            rec_cluster_res(note='-' + linkage + "_n=" + str(n_cluster))
    elif method == 'Kmeans':
        model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=20, random_state=28)
        pred_y = model.fit_predict(Xn)
        rec_cluster_res(note="n=" + str(n_cluster))


if __name__ == "__main__":
    gen_fp('C-DB/SMILESstrings.csv', 'deep/morgan_fp.csv')
    gen_fp('CB-DB/SMILESstrings.csv', 'full/morgan_fp.csv')
    df = pd.read_csv('deep/morgan_fp.csv', header=None, index_col=0)

    cluster_mols("C-DB", df, method='Hier')
    cluster_mols("C-DB", df, method='Hier', n_cluster=30)
    df = pd.read_csv('full/morgan_fp.csv', header=None, index_col=0)
    cluster_mols("CB-DB", df, method='Hier')
    cluster_mols("CB-DB", df, method='Hier', n_cluster=30)
