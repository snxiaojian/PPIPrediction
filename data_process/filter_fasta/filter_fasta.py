import os
from Bio import SeqIO
import sys
import numpy

sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record, gene_from_record, fasta_folder_from_feature_type
from go_rgcn.GOAParser import GOAParser
from ppi_network.static_args import *

def data_is_complete_and_correct(file, id, record):
    species = file.split(".")[0]

    contact_map_file = "./data/contact_map/" + species + "/" + id + ".npz"
    if not os.path.exists(contact_map_file):
        return False
    
    if not data_has_pssm(file, id, record):
        return False
    
    len1 = len(record.seq._data)
    len2 = numpy.load(contact_map_file)["contact"].shape[0]
 
    return len1 == len2

def data_has_pssm(file, id, record):
    species = file.split(".")[0]
    
    pssm_file = "./data/pssm/" + species + "/" + id + ".pssm"
    if not os.path.exists(pssm_file):
        print("dont' have pssm")
        return False
    acids = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
    if not set(record.seq._data).issubset(acids):
        return False
    return True

def start_filter(feature_type):
    input_fasta_folder = "./data/input/"
    fasta_filtered_folder = fasta_folder_from_feature_type(feature_type)
    if not os.path.exists(fasta_filtered_folder):
        os.mkdir(fasta_filtered_folder)
    files = get_fasta_names_from_folder(input_fasta_folder)
    anotations = GOAParser.parsed_annotation()
    for file in files:
        print("filtering " + file)
        species_records = []
        with open(input_fasta_folder + file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                id = id_from_record(record)
                if feature_type == feature_type_go_graph_pssm:
                    gene = gene_from_record(record)
                    has_anotation = id in anotations or gene in anotations
                    if data_is_complete_and_correct(file, id, record) and has_anotation:
                        species_records.append(record)
                elif feature_type == feature_type_graph_pssm:
                    if data_is_complete_and_correct(file, id, record):
                        species_records.append(record)
                elif feature_type == feature_type_residue_pssm:
                    if data_has_pssm(file, id, record):
                        species_records.append(record)
                else:
                    raise Exception("feature_type not supported: %s" % feature_type)

        with open(fasta_filtered_folder + file, 'w') as f:
            SeqIO.write(species_records, f, 'fasta')

if __name__ == "__main__":
    start_filter(feature_type=feature_type_residue_pssm)
