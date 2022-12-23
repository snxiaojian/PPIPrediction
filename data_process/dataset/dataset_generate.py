from __future__ import annotations
from collections import defaultdict
from Bio import SeqIO
import random
import csv
import os
import math
import sys
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record, fasta_folder_from_feature_type
from data_process.biogrid.BioGridParser import BioGridParser
from ppi_network.static_args import *

def recordIDs_for_file(file):
    recordIDs = []
    with open(file, 'r') as f:
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
                if len(positive_ppi_item) % 10000 == 0:
                    print("generating positive ppi index: " + str(len(positive_ppi_item)))
    negative_ppi_count = int(len(positive_ppi_item) * ratio)
    negative_ppi_item = []

    if negative_ppi_count == 0:
        return positive_ppi_item, negative_ppi_item
    for index in range(negative_ppi_count):
        pairs = random.sample(recordIDs, 2)
        ppi1 = pairs[0]
        ppi2 = pairs[1]
        if ppi2 in mutual_ppi_dict[ppi1]:
            continue
        negative_ppi_item.append([ppi1, ppi2, 0])
        mutual_ppi_dict[ppi1][ppi2] = 0
        mutual_ppi_dict[ppi2][ppi1] = 0
        if index % 10000 == 0:
            print("generating negative ppi index: " + str(index))
    return positive_ppi_item, negative_ppi_item
    

def negative_ratio_for_species(species, ppi_count):
    file = "./data/input/" + species + ".fasta"
    if not os.path.exists(file):
        raise ValueError("Species not supported")
    protein_num = len([1 for line in open(file) if line.startswith(">")])
    negative_ppi_ratio =  math.pow(protein_num, 2) / (ppi_count * 2 * 1000 * 3)
    return negative_ppi_ratio

def write_ppi_to_tsv(ppi_items, type):
    file = "./data/dataset/" + type + "_ppi.tsv"
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(ppi_items)
        print("appending to " + type + ".tsv")

def write_infomation_to_file(information, file):
    with open(file, 'w') as f:
        f.writelines(information)
        print("appending to " + file)

def generate_dateset(feature_type):
    folder = fasta_folder_from_feature_type(feature_type)
    files = get_fasta_names_from_folder(folder)
    positive_ppi = []
    negative_ppi = []
    informations = []
    for file in files:
        species = file.split(".")[0]
        if species not in train_species:
            print("ignoring " + file)
            continue
        print("processing " + file)
        biogrid_network, ppi_count = BioGridParser.parsed_network(species=species)
        information = "total biogrid: " + species + " ppi count: " + str(ppi_count) + "\n"
        print(information)
        informations.append(information)
        
        ratio = negative_ratio_for_species(species, ppi_count)
        ratio = min(ratio, 7)
        information = "negative ratio for " + species + ": " + str(ratio) + "\n"
        print(information)
        informations.append(information)
        recordIDs = recordIDs_for_file(folder+"/"+file)

        positive_ppi_in_species, negative_ppi_in_species = generate_ppi(ratio, biogrid_network, recordIDs)
        information = "positive ppi count for " + species + ": " + str(len(positive_ppi_in_species)) + "\n"
        print(information)
        informations.append(information)
        information = "negative ppi count for " + species + ": " + str(len(negative_ppi_in_species)) + "\n"
        print(information)
        informations.append(information)
        positive_ppi.extend(positive_ppi_in_species)
        negative_ppi.extend(negative_ppi_in_species)
    write_ppi_to_tsv(positive_ppi+negative_ppi, "whole_"+ feature_type)
    
    train_test_ratio = 0.8
    random.shuffle(positive_ppi)
    random.shuffle(negative_ppi)
    positive_ppi_count = len(positive_ppi)
    split_in_positive = int(positive_ppi_count*train_test_ratio)
    split_in_negative = int(len(negative_ppi) - positive_ppi_count * (1-train_test_ratio))
    
    train_positive_ppi = positive_ppi[:split_in_positive]
    train_negative_ppi = negative_ppi[:split_in_negative]
    test_positive_ppi = positive_ppi[split_in_positive:]
    test_negative_ppi = negative_ppi[split_in_negative:]
    
    train_dataset = train_positive_ppi + train_negative_ppi
    test_dataset = test_positive_ppi + test_negative_ppi
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    informations.append("train positive ppi count: " + str(len(train_positive_ppi)) + "\n")
    informations.append("train negative ppi count: " + str(len(train_negative_ppi)) + "\n")
    informations.append("test positive ppi count: " + str(len(test_positive_ppi)) + "\n")
    informations.append("test negative ppi count: " + str(len(test_negative_ppi)) + "\n")
    
    write_infomation_to_file(informations, "./data/dataset/informations_" + feature_type + ".txt")
    write_ppi_to_tsv(train_dataset, "train_"+ feature_type)
    write_ppi_to_tsv(test_dataset, "test_"+ feature_type)
    
if __name__ == "__main__":
    generate_dateset(feature_type=feature_type_residue_pssm)
    # generate_dateset(feature_type=feature_type_go_graph_pssm)
    # generate_dateset(feature_type=feature_type_graph_pssm)