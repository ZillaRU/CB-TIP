import os
import random
import numpy as np
import sys

'''
Usage: 
nohup python3 cmap_gen.py dataset_name  gpu_idx >/dev/null 2>&1 &
'''


def pconsc4Prediction(dataset):
    pre_path = "/data/drugbank_prot/data/"
    model = pconsc4.get_pconsc4()
    aln_dir = pre_path+dataset+'/hhfilter'
    output_dir = pre_path+dataset+'pconsc4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(aln_dir)
    random.shuffle(file_list)
    inputs = []
    outputs = []
    sudocmd('chmod -R u+r+w '+output_dir)
    for file in file_list:
        input_file = os.path.join(aln_dir, file)
        output_file = os.path.join(output_dir, file.split('.a3m')[0] + '.npy')
        if os.path.exists(output_file):
            continue
        inputs.append(input_file)
        outputs.append(output_file)
        try:
            print('process', input_file)
            pred = pconsc4.predict(model, input_file)
            print(input_file, ' prediction finished')
            np.save(output_file, pred['cmap'], allow_pickle=True)
            print(output_file, 'over.')
        except:
            print(output_file, 'error.')


def sudocmd(cmd):
    print(cmd)
    rzy_pass = 'xxxxxx'
    os.system("echo %s|sudo -S %s " % (rzy_pass, cmd))


if __name__ == '__main__':
    sudocmd('chmod -R u+r+w /data/drugbank_prot/data')
    import pconsc4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
    pconsc4Prediction(sys.argv[1])
