import torch
import os
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader


num_pick = 50
loader = ProteinFeatureLoader(pick_num= num_pick).default_loader
    
def process(para_list):
    pid = para_list[0]
    index = para_list[1]
    G_residue, go_embedding, norm_picked_pssm, indexes = loader(pid)
    save_tensor(pid,G_residue,'G_residue',num_pick)
    save_tensor(pid,go_embedding,'go_embedding',num_pick)
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
    
        
if __name__ == "__main__":
    folder = "./data/filtered_input/"
    files = get_fasta_names_from_folder(folder)
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
    para_list = []
    for index, pid in enumerate(recordIDs):
        #   para_list.append([pid, index])
          process([pid, index])
    # with ProcessPoolExecutor(max_workers=32) as pool:
    #     pool.map(process,para_list)
    