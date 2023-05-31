import os
from Bio import SeqIO
import sys
sys.path.append("./")
from ppi_network.static_args import *

def get_fasta_names_from_folder(folder):
    return [name for name in os.listdir(folder) if name.endswith(".fasta")]

def id_from_record(record):
    if record.id.find("|") == -1:
        return record.id
    return record.id.split('|')[1]

def gene_from_record(record):
    if record.description.find("GN=") == -1:
        return id_from_record(record)
    return record.description.split('GN=')[1].split(" ")[0]

def fasta_folder_from_feature_type(feature_type):
    if feature_type == feature_type_go_graph_pssm:
        return "./data/filtered_input_with_go/"
    elif feature_type == feature_type_graph_pssm:
        return "./data/filtered_input_no_go/"
    elif feature_type == feature_type_residue_pssm:
        return "./data/filtered_input_only_pssm/"
    else:
        raise Exception("feature_type not supported: %s" % feature_type)

def records_from_filtered_input(feat_type):
    fasta_folder = fasta_folder_from_feature_type(feat_type)
    records = {}
    species_dict = {}
    files = get_fasta_names_from_folder(fasta_folder)
    for file in files:
        species = file.split(".")[0]
        for record in SeqIO.parse(fasta_folder + file, "fasta"):
            records[id_from_record(record)] = record
            species_dict[id_from_record(record)] = species
    return records, species_dict

def records_from_fasta_file(file):
    records = {}
    for record in SeqIO.parse(file, "fasta"):
        records[id_from_record(record)] = record
    return records

def ids_from_fasta_file(file):
    ids = []
    for record in SeqIO.parse(file, "fasta"):
        ids.append(id_from_record(record))
    return ids