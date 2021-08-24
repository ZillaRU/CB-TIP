import os
import sys
'''
Usage: 
sudo nohup python msa_aln_gen.py dataset_name >/dev/null 2>&1 &
'''


def HHblitsMSA(bin_path, db_path, input_dir, output_dir):
    for fas_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, fas_file)
        output_file = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.hhr')  # igore
        output_file_a3m = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.a3m')
        if os.path.exists(output_file) and os.path.exists(output_file_a3m):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        output_file_a3m = output_file_a3m.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' -maxfilt 100000 -realign_max 100000 -d ' + db_path + ' -all -B 100000 -Z 100000 -n 3 -e 0.001 -i ' + process_file + ' -o ' + output_file + ' -oa3m ' + output_file_a3m + ' -cpu 16'
        print(cmd)
        os.system(cmd)


def HHfilter(bin_path, input_dir, output_dir):
    file_prefix = []
    for file in os.listdir(input_dir):
        if 'a3m' not in file:
            continue
        temp_prefix = file.split('.a3m')[0]
        if temp_prefix not in file_prefix:
            file_prefix.append(temp_prefix)
    for msa_file_prefix in file_prefix:
        file_name = msa_file_prefix + '.a3m'
        process_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' -id 90 -i ' + process_file + ' -o ' + output_file
        print(cmd)
        os.system(cmd)


def reformat(bin_path, input_dir, output_dir):
    # print('reformat')
    for a3m_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, a3m_file)
        output_file = os.path.join(output_dir, a3m_file.split('.a3m')[0] + '.fas')
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' ' + process_file + ' ' + output_file + ' -r'
        print(cmd)
        os.system(cmd)


def sudocmd(cmd):
    print(cmd)
    rzy_pass = '123456'
    os.system("echo %s|sudo -S %s " % (rzy_pass, cmd))


def convertAlignment(bin_path, input_dir, output_dir):
    # print('convertAlignment')
    for fas_file in os.listdir(input_dir):
        process_file = input_dir + '/' + fas_file
        output_file = output_dir + '/' + fas_file.split('.fas')[0] + '.aln'
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = 'python ' + bin_path + ' ' + process_file + ' fasta ' + output_file
        print(cmd)
        os.system(cmd)


def alnFilePrepare(dataset):
    import json
    from collections import OrderedDict
    print('aln file prepare for '+dataset+'...')
    pre_path = "/data/drugbank_prot/data/"
    seq_dir = pre_path+dataset+'/seq'
    msa_dir = pre_path+dataset+'/msa'
    filter_dir = pre_path+dataset+'/hhfilter'
    reformat_dir = pre_path+dataset+'/reformat'
    aln_dir = pre_path+dataset+'/aln'
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    if not os.path.exists(msa_dir):
        os.makedirs(msa_dir)
    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)
    if not os.path.exists(reformat_dir):
        os.makedirs(reformat_dir)
    if not os.path.exists(aln_dir):
        os.makedirs(aln_dir)
    HHblits_bin_path = '/home/rzy/HHblits/bin/hhblits'  # HHblits bin path
    HHblits_db_path = '/data/HHDB/UniRef30_2020_03'  # hhblits dataset for msa
    HHfilter_bin_path = '/home/rzy/HHblits/bin/hhfilter'  # HHfilter bin path
    reformat_bin_path = '/home/rzy/HHblits/scripts/reformat.pl'  # reformat bin path
    convertAlignment_bin_path = '/home/rzy/CCMpred/scripts/convert_alignment.py'  # ccmpred convertAlignment bin path
    sudocmd('chmod -R u+r /data/HHDB')
    # check the programs used for the script
    if not os.path.exists(HHblits_bin_path):
        raise Exception('Program HHblits was not found. Please specify the run path.')
    os.system('chmod -R u+r+x ' + HHblits_bin_path)
    if not os.path.exists(HHfilter_bin_path):
        raise Exception('Program HHfilter was not found. Please specify the run path.')
    os.system('chmod -R u+r+x ' + HHfilter_bin_path)
    if not os.path.exists(reformat_bin_path):
        raise Exception('Program reformat was not found. Please specify the run path.')
    os.system('chmod u+r+x ' + reformat_bin_path)
    if not os.path.exists(convertAlignment_bin_path):
        raise Exception('Program convertAlignment was not found. Please specify the run path.')
    os.system('chmod u+r+x ' + HHfilter_bin_path)
    # seq_format(proteins, seq_dir)
    HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
    HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
    reformat(reformat_bin_path, filter_dir, reformat_dir)
    convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)
    print('aln file prepare over for '+dataset)


if __name__ == '__main__':
    sudocmd('chmod -R u+r+w /data/drugbank_prot/data')
    alnFilePrepare(sys.argv[1])
