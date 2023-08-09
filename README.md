# CB-TIP — *T*ype-aware *I*nteraction *P*rediction for both *C*hemical and *B*iotech drugs

## Overview
CB-TIP is a graph-based multi-relational link prediction framework tailored for DDI prediction, which is useful for providing prescript tips to clinicians. See [our paper](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbad271/7233058) for details on the framework.

## Running the code 

### Requirements
You need to create a CONDA environment with the command `conda env create -f torch17_chembiotip.yaml`.

### Datasets
We provide two preprocessed real-world datasets _C-DB_ and _CB-DB_ (path: `data`) used in our paper.
To reproduce the results reported in our paper, there is no need to construct the intra-view graphs from the raw linear expressions of molecules.
We share the archives of LMDB files storing the built intra-view graphs (path: `data/DATASET_NAME/lmdb_files`).

*If you want to run this code on your own dataset,
1. Generate the residue contact maps for large molecules (please refer to Section 7.1 and the files i)`utils/msa_aln_gen.py` ii) `utils/cmap_gen.py`).
2. Organize your data as the form like `data/CB-DB/`.
3. Construct intra-view graphs and save them into LMDB files with the code `utils/generate_intra_graph_db.py`.

### Intra-view sample pair generation
1. Run `data/calc_fp.py` to generate fingerprints for chemical drugs and divide them into clusters.
2. Run `data/construct_pp_nn.py` to similar pairs and dissimilar pairs of chemical drugs.

### Start training
To run the training process under default settings, use the following command:
```
$ python train_main.py -d DATASET_NAME -sp SPLIT
```
options for DATASET_NAME: C-DB, CB-DB
Negative samples: You can use the negative samples in data/DATASET_NAME/ddi_neg.csv, or generate negative samples using the code `utils/sample_neg_split.py`.
SPLIT: You can split the positive/ negative samples into training/ validation/ test sets using the code `utils/sample_neg_split.py`. Please remember to update the data path in the function `dd_dt_tt_build_inter_graph_from_links` in `utils/hete_data_utils.py`.

For more customized settings, please refer to `utils/arg_parser.py`.

### BibTex of our CB-TIP
If you use CB-TIP in your research, please use the following BibTeX entry. 📣 Thank you!
```bibtex
@article{10.1093/bib/bbad271,
    author = {Ru, Zhongying and Wu, Yangyang and Shao, Jinning and Yin, Jianwei and Qian, Linghui and Miao, Xiaoye},
    title = "{A dual-modal graph learning framework for identifying interaction events among chemical and biotech drugs}",
    journal = {Briefings in Bioinformatics},
    pages = {bbad271},
    year = {2023},
    month = {07},
    issn = {1477-4054},
    doi = {10.1093/bib/bbad271}
}
```
