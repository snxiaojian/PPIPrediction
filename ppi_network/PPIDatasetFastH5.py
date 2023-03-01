
from torch.utils.data import Dataset
import torch
import re
import sys
sys.path.append("./")
from data_process.tensor_to_disk.save_dateset_to_h5file import key, h5file
from ppi_network.static_args import *
from data_process.util import records_from_filtered_input

import h5py

def fast_no_go_collate(samples):
    residue_with_pssm, residue_with_pssm2, labels = map(list, zip(*samples))
    return residue_with_pssm, residue_with_pssm2, torch.FloatTensor(labels)            

class PPIDatasetFastH5(Dataset):
    def __init__(self,type, target_species = None):
        
        super(PPIDatasetFastH5,self).__init__()
        _, species_dict = records_from_filtered_input(feature_type_residue_pssm)
        self.species_dict = species_dict
        self.default_loader = self.disk_loader
        ppi_items=[]
        with open('./data/dataset/'+type + "_" + feature_type_residue_pssm +'_ppi.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split(' |\t',line)
                species = self.species_dict[words[0]]
                if target_species is None or species == target_species:
                    ppi_items.append((words[0],words[1],int(words[2])))
                
        self.ppi_items = ppi_items

    def __getitem__(self, index):
        p1,p2, label = self.ppi_items[index]
        residue_features = self.default_loader(p1) 
        residue_features2 = self.default_loader(p2)
        return residue_features, residue_features2, label

    def __len__(self):
        return len(self.ppi_items)
    
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