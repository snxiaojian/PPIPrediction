import torch
import torch.nn.functional as F
import random
from ppi_network.static_args import *

class PSSFastPPIModel(torch.nn.Module):

    def __init__(self, device, pick_num):
        super(PSSFastPPIModel,self).__init__()
        self.device = device
        self.pick_num = pick_num
        self.graph_embedding_size = residue_embedding_size
        self.pssm_embedding_size = pick_num * 20
        self.drop = 0.2

        
        # final residue size before combine acid
        self.residue_out_dim = 64
        self.graph_pssm_output_dim = 64

        # gcn
        
        self.relu = torch.nn.ReLU()
        self.fc_g1 = torch.nn.Linear((self.graph_embedding_size + pssm_size), self.residue_out_dim)
        self.fc_g2 = torch.nn.Linear(self.residue_out_dim * self.pick_num, self.graph_pssm_output_dim)

        self.dropout = torch.nn.Dropout(self.drop)

        # combined layers
        self.fc = torch.nn.Linear( self.graph_pssm_output_dim*2, 32)
        self.out = torch.nn.Linear(32, 1)

    # input1 input2
    def forward(self, residue_features, residue_features2):
        # protein1
        protein_feature = self.forward_part(residue_features)
        protein_feature2 = self.forward_part(residue_features2)
        if bool(random.getrandbits(1)):
            feature = torch.cat((protein_feature2, protein_feature), dim=1)
        else:
            feature = torch.cat((protein_feature, protein_feature2), dim=1)
        
        x = self.fc(feature)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        output = torch.sigmoid(out)
        return output
    
    def forward_part(self, residue_features):
        batch_size = residue_features.shape[0]        
        g_feature = self.relu(self.fc_g1(residue_features))
        g_feature = g_feature.reshape(batch_size, -1)
        g_feature = self.relu(self.fc_g2(g_feature))
        return g_feature