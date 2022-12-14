import torch
import os
from Bio import SeqIO
import sys
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader
    
def process(pid, index, num_pick, has_go, loader):
    if has_go:
        G_residue, go_embedding, norm_picked_pssm, indexes = loader(pid)
        save_tensor(pid,G_residue,'G_residue',num_pick)
        save_tensor(pid,go_embedding,'go_embedding',num_pick)
        save_tensor(pid,norm_picked_pssm,'norm_picked_pssm',num_pick)
        save_tensor(pid,indexes,'indexes',num_pick)
    else:
        G_residue, norm_picked_pssm, indexes = loader(pid)
        save_tensor(pid,G_residue,'G_residue',num_pick)
        save_tensor(pid,norm_picked_pssm,'norm_picked_pssm',num_pick)
        save_tensor(pid,indexes,'indexes',num_pick)
    # if index % 10 == 0:
    print("process: " + str(index) + " " + pid)
    filename = tensor_filename(pid,"G_residue",num_pick)
    if not os.path.exists(filename):
        raise Exception("not exist: " + filename)
    
def tensor_filename(pid,type,num_pick):
    return "./data/tensor/" + "pid"+pid+"type"+type+"picknum"+str(num_pick)+'.pt'
    
def save_tensor(pid,tensor,type,num_pick):
    filename = tensor_filename(pid,type,num_pick)
    if not os.path.exists(filename):
        torch.save(tensor,filename)
    
def write_dataset_to_disk(num_pick, has_go):
    if has_go:
        folder = "./data/filtered_input_with_go/"
    else:
        folder = "./data/filtered_input_no_go/"
        
    files = get_fasta_names_from_folder(folder)
    
    loader = ProteinFeatureLoader(pick_num=num_pick, has_go=has_go).default_loader
    
    output_folder = "./data/tensor/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    recordIDs = []
    for file in files:
        print("processing " + file)
        with open(folder + file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                id = id_from_record(record)
                recordIDs.append(id)
    for index, pid in enumerate(recordIDs):
          process(pid, index, num_pick, has_go, loader)
    
if __name__ == "__main__":
    num_pick = 100
    write_dataset_to_disk(num_pick=num_pick, has_go=False)
    # write_dataset_to_disk(has_go=True)
    