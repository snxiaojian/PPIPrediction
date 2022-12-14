import os
from Bio import SeqIO

def get_fasta_names_from_folder(folder):
    return [name for name in os.listdir(folder) if name.endswith(".fasta")]

def id_from_record(record):
    return record.id.split('|')[1]
def gene_from_record(record):
    if record.description.find("GN=") == -1:
        return id_from_record(record)
    return record.description.split('GN=')[1].split(" ")[0]

def records_from_filtered_input(has_go):
    if has_go:
        fasta_filtered_folder = "./data/filtered_input_with_go/"
    else:
        fasta_filtered_folder = "./data/filtered_input_no_go/"
    records = {}
    species_dict = {}
    for file in get_fasta_names_from_folder(fasta_filtered_folder):
        species = file.split(".")[0]
        for record in SeqIO.parse(fasta_filtered_folder + file, "fasta"):
            records[id_from_record(record)] = record
            species_dict[id_from_record(record)] = species
    return records, species_dict