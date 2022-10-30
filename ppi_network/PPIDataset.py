from torch.utils.data import DataLoader,Dataset
import torch
import re
import ProteinFeatureLoader

def collate(samples):

    G_residue, G_go, pssm, G_residue2, G_go2, pssm2, labels = map(list, zip(*samples))
    return G_residue, G_go, pssm, G_residue2, G_go2, pssm2,torch.tensor(labels)        

class PPIDataset(Dataset):
    def __init__(self,type, loader=ProteinFeatureLoader.default_loader):
        
        super(PPIDataset,self).__init__()
        
        ppi_items=[]
        with open('./data/'+type+'_ppi.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split(' |\t',line)
                ppi_items.append((words[0],words[1],int(words[2])))
                
        self.ppi_items = ppi_items
        self.transform = None
        self.target_transform = None
        self.loader = loader        

    def __getitem__(self, index):
        p1,p2, label = self.ppi_items[index]
        G_residue, G_go, pssm = self.loader(p1) 
        G_residue2, G_go2, pssm2 = self.loader(p2)
        return G_residue, G_go, pssm, G_residue2, G_go2, pssm2, label

    def __len__(self):
        return len(self.ppi_items)