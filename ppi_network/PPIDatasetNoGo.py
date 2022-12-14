
from torch.utils.data import DataLoader,Dataset
import torch
import re
import sys
sys.path.append("./")
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader
from data_process.tensor_to_disk.save_dataset_tensor_to_disk import tensor_filename

def collate(samples):
    G_residue, pssm, indexes, G_residue2, pssm2, indexes2, labels = map(list, zip(*samples))
    return G_residue, pssm, indexes, G_residue2, pssm2, indexes2, torch.FloatTensor(labels)            

class PPIDatasetNoGo(Dataset):
    def __init__(self,type, pick_num):
        
        super(PPIDatasetNoGo,self).__init__()
        # self.default_loader = ProteinFeatureLoader(pick_num=pick_num).default_loader
        self.default_loader = self.disk_loader
        self.pick_num = pick_num
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
        G_residue, pssm, indexes = self.default_loader(p1) 
        G_residue2, pssm2, indexes2 = self.default_loader(p2)
        return G_residue, pssm, indexes, G_residue2, pssm2, indexes2, label

    def __len__(self):
        return len(self.ppi_items)
    
    def disk_loader(self,pid):
        G_residue_file =  tensor_filename(pid,'G_residue',self.pick_num)
        G_residue = torch.load(G_residue_file)
        
        norm_picked_pssm_file = tensor_filename(pid,'norm_picked_pssm',self.pick_num)
        norm_picked_pssm = torch.load(norm_picked_pssm_file)
        
        indexes_file = tensor_filename(pid,'indexes',self.pick_num)
        indexes = torch.load(indexes_file)
        return G_residue, norm_picked_pssm, indexes 