from __future__ import annotations
from collections import defaultdict
import sys
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record
from data_process.biogrid.BioGridParser import BioGridParser
from Bio import SeqIO
import random
import csv
import numpy

def recordIDs_for_file(file):
    fasta_file = "./data/filtered_input/" + file
    recordIDs = []
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            recordIDs.append(id_from_record(record))
    return recordIDs

def generate_ppi(ratio, biogrid_network, recordIDs):
    positive_ppi_item = []
    mutual_ppi_dict = defaultdict(dict)
    for ppi1, ppi2dict in biogrid_network.items():
        for ppi2, value in ppi2dict.items():
            if ppi1 in recordIDs and ppi2 in recordIDs:
                if ppi2 in mutual_ppi_dict[ppi1]:
                    continue
                positive_ppi_item.append([ppi1, ppi2, 1])
                mutual_ppi_dict[ppi1][ppi2] = 1
                mutual_ppi_dict[ppi2][ppi1] = 1

    negative_ppi_count = len(positive_ppi_item) * ratio
    negative_ppi_item = []
    for index in range(negative_ppi_count):
        pairs = random.sample(recordIDs, 2)
        ppi1 = pairs[0]
        ppi2 = pairs[1]
        if ppi2 in mutual_ppi_dict[ppi1]:
            continue
        negative_ppi_item.append([ppi1, ppi2, 0])
        mutual_ppi_dict[ppi1][ppi2] = 0
        mutual_ppi_dict[ppi2][ppi1] = 0
    return positive_ppi_item, negative_ppi_item
    

def safe_ratio_for_species(species):
    if species == "human":
        return 7
    elif species == "fly":
        return 5
    elif species == "yeast":
        return 0.1
    elif species == "arabidopsis":
        return 40
    else:
        raise ValueError("Species not supported")

def write_ppi_to_tsv(ppi_items, type):
    file = "./data" + type + "_ppi.tsv"
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(ppi_items)
        print("appending to " + species + ".tsv")

if __name__ == "__main__":
    files = get_fasta_names_from_folder("./data/filtered_input")
    positive_ppi = []
    negative_ppi = []
    biogrid_network = BioGridParser.parsed_network()
    for file in files:
        species = file.split(".")[0]
        ratio = safe_ratio_for_species(species)
        recordIDs = recordIDs_for_file(file)

        positive_ppi_in_species, negative_ppi_in_species = generate_ppi(ratio, biogrid_network, recordIDs)
        positive_ppi.extend(positive_ppi_in_species)
        negative_ppi.extend(negative_ppi_in_species)
    write_ppi_to_tsv(positive_ppi+negative_ppi, "whole")
    ratio = 0.8
    random_positive_ppi = numpy.random.shuffle(positive_ppi)
    random_negative_ppi = numpy.random.shuffle(negative_ppi)
    train_positive_ppi = random_positive_ppi[:int(len(random_positive_ppi)*ratio)]
    train_negative_ppi = random_negative_ppi[:int(len(random_negative_ppi)*ratio)]
    test_positive_ppi = random_positive_ppi[int(len(random_positive_ppi)*ratio):]
    test_negative_ppi = random_negative_ppi[int(len(random_negative_ppi)*ratio):]
    write_ppi_to_tsv(train_positive_ppi+train_negative_ppi, "train")
    write_ppi_to_tsv(test_positive_ppi+test_negative_ppi, "test")

    
    