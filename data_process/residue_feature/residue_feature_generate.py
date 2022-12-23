from email.policy import default
from operator import mod
import sys
from Bio import SeqIO
from gensim.models import Word2Vec
import json
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder
from ppi_network.static_args import *

def load_sequence_data():
    folder = "./data/input/"
    files = get_fasta_names_from_folder(folder)
    sequences = []
    for file in files:
        print("get sequence from " + file)
        with open(folder + file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                 sequences.append(" " + record.seq._data + " ")
    return sequences

def generate_triple_feature(feature_size, epochs):
    sequences = load_sequence_data()
    k = 3
    triple_AA_list = [[seq[i-k//2:i+k//2+1] for i in range(k//2,len(seq)-k//2)] for seq in sequences]
    triples_set = set()
    triple_count = 0
    for sentence in triple_AA_list:
        for triple in sentence:
            if triple not in triples_set:
                triples_set.add(triple)
    triple_count = len(triples_set)
    print("triple count: " + str(triple_count))
    feat_size = feature_size
    model = Word2Vec(triple_AA_list, window=13, vector_size=feat_size, workers=32, sg=1, min_count=1, epochs=epochs)
    seq2vec_result = {}
    model_result = model.wv
    for triple in triples_set:
        if triple in model_result:
            seq2vec_result[triple] = [float(value) for value in model_result[triple]]
        else:
            print(triple,'not in training docs...')
    return seq2vec_result

def save_triple_feature_dict_with_json(jsonObject):
    with open('./data/residue_feature/triple_feature.json', 'w') as f:
        json.dump(jsonObject, f)

if __name__ == '__main__':
    triple_feature = generate_triple_feature(feature_size=residue_embedding_size-27, epochs=10)
    save_triple_feature_dict_with_json(triple_feature)
    
