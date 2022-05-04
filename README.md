# ChemBioTIP — *T*ype-aware *I*nteraction *P*rediction for both *Chem*ical drugs and *Bio*tech drugs

## Overview
ChemBioTIP is a graph-based multi-relational link prediction framework tailored for DDI prediction, which is useful for providing prescript tips to clinicians. See our paper(link: ) for details on the framework.

## Running the code 

### Requirements
You need to create a CONDA environment with the command `conda env create -f torch17_chembiotip.yaml`.

### Datasets
We provide two preprocessed real-world datasets _C-DB_ and _CB-DB_ (path: `data`) used in our paper.
To reproduce the results reported in our paper, there is no need to construct the intra-view graphs from the raw linear expressions of molecules.
We share the archives of LMDB files storing the built intra-view graphs (path: `data/DATASET_NAME/lmdb_files`).

*If you want to run this code on your own dataset,
1. Generate the residue contact maps for large molecules (please refer to Section 7.1 and `utils/msa_aln_gen.py`).
2. Organize your data as the form like `data/CB-DB/`.
3. Construct intra-view graphs and save them into LMDB files with the code `utils/generate_intra_graph_db.py`.

### Start training
To run the training process under default settings, use the following command:
```
$ python train_main.py -d DATASET_NAME -sp SPLIT
```
options for DATASET_NAME: C-DB, CB-DB
Negative samples: You can use the negative samples in data/DATASET_NAME/ddi_neg.csv, or generate negative samples using the code `utils/sample_neg_split.py`.
SPLIT: You can split the positive/ negative samples into training/ validation/ test sets using the code `utils/sample_neg_split.py`. Please remember to update the data path in the function `dd_dt_tt_build_inter_graph_from_links` in `utils/hete_data_utils.py`.

For more customized settings, please refer to `utils/arg_parser.py` and Section 7 in our paper.
