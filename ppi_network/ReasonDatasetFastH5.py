
from torch.utils.data import Dataset
import torch
import re
import sys
sys.path.append("./")
from data_process.tensor_to_disk.save_dateset_to_h5file import key, h5file
from ppi_network.static_args import *
from data_process.util import *

import h5py

def fast_no_go_collate(samples):
    residue_with_pssm, residue_with_pssm2 = map(list, zip(*samples))
    return residue_with_pssm, residue_with_pssm2           

def predict_protein(n: int, x: int):
    total = n * (n - 1) / 2
    
    if x < 1 or x > total:
        return None
    
    row = int((2 * x - 0.25)**0.5 + 0.5)
    col = x - row * (row - 1) // 2
    
    prot_A = n - row
    prot_B = prot_A + col
    
    return (prot_A, prot_B)

def pid_pairs_for(ids, predicting_index):
    proteinNum = len(ids)
    index1, index2 = predict_protein(proteinNum, predicting_index + 1)
    pid1 = ids[index1-1]
    pid2 = ids[index2-1]
    return pid1, pid2

class ReasonDatasetFastH5(Dataset):
    def __init__(self,species, startIndex):
        super(ReasonDatasetFastH5,self).__init__()
        self.species = species
        self.startIndex = startIndex
        self.default_loader = self.disk_loader
        folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
        file = folder + "expressed_" + species + ".fasta"
        self.records = records_from_fasta_file(file)
        self.ids = ids_from_fasta_file(file)
        self.proteinNum = len(self.ids)
        self.lenth = int((self.proteinNum * (self.proteinNum - 1))/2) - startIndex
    
    def __getitem__(self, index):
        index = index + self.startIndex
        pid1, pid2 = pid_pairs_for(self.ids, index)
        residue_features = self.default_loader(pid1) 
        residue_features2 = self.default_loader(pid2)
        return residue_features, residue_features2

    def __len__(self):
        return self.lenth
    
    def open_hdf5(self):
        self.h5file = h5py.File(h5file, 'r')
        
    def disk_loader(self,pid):   
        if not hasattr(self, 'h5file'):
            self.open_hdf5()     
        name =  key(pid,feature_type_residue_pssm,pick_num_fast)
        value = self.h5file[self.species][name][:]
        tensor = torch.Tensor(value)
        return tensor