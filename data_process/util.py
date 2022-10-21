import os
from Bio import SeqIO

def get_fasta_names_from_folder(folder):
    return [name for name in os.listdir(folder) if name.endswith(".fasta")]

def id_from_record(record):
    return record.id.split('|')[1]