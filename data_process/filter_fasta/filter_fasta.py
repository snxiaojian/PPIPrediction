import os
from Bio import SeqIO
import sys

sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record, gene_from_record
from go_rgcn.GOAParser import GOAParser

def data_is_complete(file, id):
    species = file.split(".")[0]
    contact_map_file = "./data/alphafolddb/" + species + "/" + "AF-" + id + "-F1-model_v3.cif"
    pssm_file = "./data/pssm/" + species + "/" + id + ".pssm"
    contact_map_exist = os.path.exists(contact_map_file)
    pssm_exist = os.path.exists(pssm_file) 
    return contact_map_exist and pssm_exist

if __name__ == "__main__":
    fasta_folder = "./data/input/"
    fasta_filtered_folder = "./data/filtered_input/"
    files = get_fasta_names_from_folder(fasta_folder)
    anotations = GOAParser.parsed_annotation()
    for file in files:
        print("filtering " + file)
        species_records = []
        with open(fasta_folder + file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                id = id_from_record(record)
                gene = gene_from_record(record)
                has_anotation = id in anotations or gene in anotations
                if data_is_complete(file, id) and has_anotation:
                    species_records.append(record)

        with open(fasta_filtered_folder + file, 'w') as f:
            SeqIO.write(species_records, f, 'fasta')
