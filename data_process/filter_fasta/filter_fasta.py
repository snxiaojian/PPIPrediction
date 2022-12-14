import os
from Bio import SeqIO
import sys
import numpy

sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record, gene_from_record
from go_rgcn.GOAParser import GOAParser

def data_is_complete_and_correct(file, id, record):
    species = file.split(".")[0]

    contact_map_file = "./data/contact_map/" + species + "/" + id + ".npz"
    if not os.path.exists(contact_map_file):
        return False
    
    pssm_file = "./data/pssm/" + species + "/" + id + ".pssm"
    if not os.path.exists(pssm_file):
        return False
    
    len1 = len(record.seq._data)
    len2 = numpy.load(contact_map_file)["contact"].shape[0]
 
    return len1 == len2

def start_filter(with_go):
    input_fasta_folder = "./data/input/"
    if with_go:
        fasta_filtered_folder = "./data/filtered_input_with_go/"
    else:
        fasta_filtered_folder = "./data/filtered_input_no_go/"
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
                if with_go:
                    gene = gene_from_record(record)
                    has_anotation = id in anotations or gene in anotations
                    if data_is_complete_and_correct(file, id, record) and has_anotation:
                        species_records.append(record)
                else:
                    if data_is_complete_and_correct(file, id, record):
                        species_records.append(record)

        with open(fasta_filtered_folder + file, 'w') as f:
            SeqIO.write(species_records, f, 'fasta')

if __name__ == "__main__":
    start_filter(with_go=True)
    start_filter(with_go=False)
