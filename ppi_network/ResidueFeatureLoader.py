import json
import csv
from Bio import SeqIO

class ResidueFeatureLoader:
    def __init__(self):
        self.triple_feature_json = load_triple_feature_json()

    def load_residue_feature(self, record):
        seq = " " + record.seq._data + " "
        k = 3
        triple_AA_list = [seq[i-k//2:i+k//2+1] for i in range(k//2,len(seq)-k//2)]
        physiochemical_feature = load_physiochemical_feature()
        onehot_feature = generate_onehot_feature()
        triple_feature = []
        for triple in triple_AA_list:
            aa = triple[1]
            feature_list = onehot_feature[aa] + physiochemical_feature[aa] + self.triple_feature_json[triple]
            triple_feature.append(feature_list)
        return triple_feature
        


def load_triple_feature_json():
    with open('./data/residue_feature/triple_feature.json', 'r') as f:
        triple_feature = json.load(f)
    return triple_feature

def load_physiochemical_feature():
    with open('./data_process/residue_feature/normalized_feature.csv') as C:
        normalized_feature=csv.reader(C)
        feature_hash = {}
        amino_acid = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        for charater in amino_acid:
            feature_hash[charater] = []
        for row in normalized_feature:
            i = 0
            for charater in amino_acid:
                feature_hash[charater] += [float(row[i])]
                i += 1
    return feature_hash

def generate_onehot_feature():
    amino_acid = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    onehot_feature = {}
    for charater in amino_acid:
        onehot_feature[charater] = [0]*20
        onehot_feature[charater][amino_acid.index(charater)] = 1
    return onehot_feature

if __name__ == '__main__':
    loader = ResidueFeatureLoader()
    file = "./data/input/human.fasta"
    with open(file, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            result = loader.load_residue_feature(record)
            print(result)
            break