import torch
import os
from Bio import SeqIO
import sys
sys.path.append("./")
from data_process.util import records_from_filtered_input
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader
from ppi_network.static_args import *
    
def process(pid, index, feature_type, loader):
    if feature_type == feature_type_go_graph_pssm:
        G_residue, go_embedding, norm_picked_pssm, indexes = loader(pid)
        save_tensor(pid,G_residue,'G_residue',pick_num_precise)
        save_tensor(pid,go_embedding,'go_embedding',pick_num_precise)
        save_tensor(pid,norm_picked_pssm,'norm_picked_pssm',pick_num_precise)
        save_tensor(pid,indexes,'indexes',pick_num_precise)
    elif feature_type == feature_type_graph_pssm:
        G_residue, norm_picked_pssm, indexes = loader(pid)
        save_tensor(pid,G_residue,'G_residue',pick_num_precise)
        save_tensor(pid,norm_picked_pssm,'norm_picked_pssm',pick_num_precise)
        save_tensor(pid,indexes,'indexes',pick_num_precise)
    elif feature_type == feature_type_residue_pssm:
        residue_with_pssm = loader(pid)
        save_tensor(pid,residue_with_pssm,'residue_with_pssm',pick_num_fast)
    else:
        raise Exception("feat_type not supported: %s" % feature_type)
    if index%100 == 0:
        print("processed %d" % index)
    
def tensor_filename(pid,type,num_pick):
    return "./data/tensor/" + "pid"+pid+"type"+type+"picknum"+str(num_pick)+'.pt'
    
def save_tensor(pid,tensor,type,num_pick):
    filename = tensor_filename(pid,type,num_pick)
    if not os.path.exists(filename):
        torch.save(tensor,filename)
        
def write_dataset_to_disk(feature_type):        
    recordIDs, species_dict = records_from_filtered_input(feature_type)

    loader = ProteinFeatureLoader(feature_type).default_loader
    print("total number of records: %d" % len(recordIDs))
    for index, pid in enumerate(recordIDs):
          process(pid, index, feature_type, loader)
    
if __name__ == "__main__":
    write_dataset_to_disk(feature_type=feature_type_residue_pssm)
    