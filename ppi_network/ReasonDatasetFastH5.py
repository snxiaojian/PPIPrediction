
from torch.utils.data import Dataset
import torch
import re
import sys
sys.path.append("./")
from data_process.tensor_to_disk.save_dateset_to_h5file import key, h5file
from ppi_network.static_args import *
from data_process.util import records_from_filtered_input, fasta_folder_from_feature_type, id_from_record

import h5py

def fast_no_go_collate(samples):
    residue_with_pssm, residue_with_pssm2 = map(list, zip(*samples))
    return residue_with_pssm, residue_with_pssm2           

class ReasonDatasetFastH5(Dataset):
    def __init__(self,species):
        super(ReasonDatasetFastH5,self).__init__()
        _, species_dict = records_from_filtered_input(feature_type_residue_pssm)
        self.species_dict = species_dict
        self.default_loader = self.disk_loader
        folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
        file = folder + species + ".fasta"
        self.ids = id_from_record(file)
        lenth = len(self.ids)
        self.lenth = (lenth * (lenth - 1))/2
    
    def __getitem__(self, index):
        # p1,p2, label = self.ppi_items[index]
        index1 = index/len(self.ids - 1)
        index2 = index - index1 * len(self.ids - 1)
        p1 = self.ids[index1]
        p2 = self.ids[index2]
        residue_features = self.default_loader(p1) 
        residue_features2 = self.default_loader(p2)
        return residue_features, residue_features2

    def __len__(self):
        return self.lenth
    
    def open_hdf5(self):
        self.h5file = h5py.File(h5file, 'r')
        
    def disk_loader(self,pid):   
        if not hasattr(self, 'h5file'):
            self.open_hdf5()     
        name =  key(pid,feature_type_residue_pssm,pick_num_fast)
        species = self.species_dict[pid]
        value = self.h5file[species][name][:]
        tensor = torch.Tensor(value)
        return tensor