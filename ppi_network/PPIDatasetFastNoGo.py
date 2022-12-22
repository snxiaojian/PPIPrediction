
from torch.utils.data import DataLoader,Dataset
import torch
import re
import sys
sys.path.append("./")
from data_process.tensor_to_disk.save_dataset_tensor_to_disk import tensor_filename
from ppi_network.ResidueFeatureLoader import ResidueFeatureLoader
from ppi_network.static_args import *

def fast_no_go_collate(samples):
    residue_with_pssm, residue_with_pssm2, labels = map(list, zip(*samples))
    return residue_with_pssm, residue_with_pssm2, torch.FloatTensor(labels)            

class PPIDatasetFastNoGo(Dataset):
    def __init__(self,type, pick_num):
        
        super(PPIDatasetFastNoGo,self).__init__()
        self.default_loader = self.disk_loader
        self.pick_num = pick_num
        self.residue_feature_loader = ResidueFeatureLoader()
        ppi_items=[]
        with open('./data/dataset/'+type+'_no_go_ppi.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split(' |\t',line)
                ppi_items.append((words[0],words[1],int(words[2])))
                
        self.ppi_items = ppi_items

    def __getitem__(self, index):
        p1,p2, label = self.ppi_items[index]
        residue_features = self.default_loader(p1) 
        residue_features2 = self.default_loader(p2)
        return residue_features, residue_features2, label

    def __len__(self):
        return len(self.ppi_items)
    
    def disk_loader(self,pid):        
        residue_with_pssm =  tensor_filename(pid,'residue_with_pssm',pick_num_fast)
        residue_with_pssm = torch.load(residue_with_pssm)
        return residue_with_pssm