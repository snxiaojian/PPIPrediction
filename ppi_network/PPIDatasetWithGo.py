from torch.utils.data import DataLoader,Dataset
import torch
import re
import sys
sys.path.append("./")
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader
from data_process.tensor_to_disk.save_dataset_tensor_to_disk import tensor_filename
from ppi_network.static_args import *

def collate(samples):
    G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, labels = map(list, zip(*samples))
    return G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, torch.FloatTensor(labels)            

class PPIDatasetWithGo(Dataset):
    def __init__(self,type):
        
        super(PPIDatasetWithGo,self).__init__()
        # self.default_loader = ProteinFeatureLoader(pick_num=pick_num).default_loader
        self.default_loader = self.disk_loader
        ppi_items=[]
        with open('./data/dataset/'+type + "_" + feature_type_go_graph_pssm +'_ppi.tsv', 'r') as fh: 	        
            for line in fh: 
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = re.split(' |\t',line)
                ppi_items.append((words[0],words[1],int(words[2])))
                
        self.ppi_items = ppi_items

    def __getitem__(self, index):
        p1,p2, label = self.ppi_items[index]
        G_residue, go_embedding, pssm, indexes = self.default_loader(p1) 
        G_residue2, go_embedding2, pssm2, indexes2 = self.default_loader(p2)
        return G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2, label

    def __len__(self):
        return len(self.ppi_items)
    
    def disk_loader(self,pid):
        G_residue_file =  tensor_filename(pid,'G_residue',pick_num_precise)
        G_residue = torch.load(G_residue_file)
        
        go_embedding_file = tensor_filename(pid,'go_embedding',pick_num_precise)
        go_embedding = torch.load(go_embedding_file)
        
        norm_picked_pssm_file = tensor_filename(pid,'norm_picked_pssm',pick_num_precise)
        norm_picked_pssm = torch.load(norm_picked_pssm_file)
        
        indexes_file = tensor_filename(pid,'indexes',pick_num_precise)
        indexes = torch.load(indexes_file)
        return G_residue, go_embedding, norm_picked_pssm, indexes 