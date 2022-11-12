from torch.utils.data import DataLoader,Dataset
import torch
import re
import sys
sys.path.append("./")
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader

def collate(samples):
    G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, labels = map(list, zip(*samples))
    return G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, torch.FloatTensor(labels)        

class PPIDataset(Dataset):
    def __init__(self,type, pick_num, device):
        
        super(PPIDataset,self).__init__()
        self.protein_feature_loader = ProteinFeatureLoader(pick_num=pick_num,device=device).default_loader
        ppi_items=[]
        with open('./data/dataset/'+type+'_ppi.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split(' |\t',line)
                ppi_items.append((words[0],words[1],int(words[2])))
                
        self.ppi_items = ppi_items

    def __getitem__(self, index):
        p1,p2, label = self.ppi_items[index]
        G_residue, go_embedding, pssm, indexes = self.protein_feature_loader(p1) 
        G_residue2, go_embedding2, pssm2, indexes2 = self.protein_feature_loader(p2)
        return G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, label

    def __len__(self):
        return len(self.ppi_items)