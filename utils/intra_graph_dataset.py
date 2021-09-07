import lmdb
from torch.utils.data import Dataset

from utils.data_utils import deserialize_small, deserialize_macro


class IntraGraphDataset(Dataset):
    def __init__(self, db_path, db_name):
        # print(db_path)
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db = self.main_env.open_db(db_name.encode())
        self.deserialize = deserialize_small if db_name == 'small_mol' else deserialize_macro
        # for key, value in self.main_env.begin(db=self.db).cursor():
        #     print (key, value)
        # self.__getitem__('0')  # test

    def __getitem__(self, mol_id):
        with self.main_env.begin(db=self.db) as txn:
            _id = mol_id.encode('ascii')
            # print(mol_id)
            graph_or_seq, size_or_graph = self.deserialize(txn.get(_id)).values()
        return mol_id, graph_or_seq, size_or_graph

    def get_nfeat_dim(self):
        with self.main_env.begin(db=self.db) as txn:
            return int.from_bytes(txn.get('node_feat_dim'.encode('ascii')), byteorder='little')

    def get_efeat_dim(self):
        with self.main_env.begin(db=self.db) as txn:
            return int.from_bytes(txn.get('edge_feat_dim'.encode('ascii')), byteorder='little')
