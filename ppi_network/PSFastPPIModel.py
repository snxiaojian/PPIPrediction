import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ppi_network.static_args import *

class PSFastPPIModel(torch.nn.Module):
    def __init__(self, device, pick_num):
        super(PSFastPPIModel, self).__init__()
        self.device = device
        self.pick_num = pick_num
        self.graph_embedding_size = residue_embedding_size
        self.pssm_embedding_size = pick_num * 20
        self.drop = 0.2

        self.residue_out_dim = 64 
        self.graph_pssm_output_dim = 256

        # Efficient layers
        self.fc_g1 = nn.Linear((self.graph_embedding_size + pssm_size), self.residue_out_dim)
        self.fc_g2 = nn.Linear(self.residue_out_dim * self.pick_num, self.graph_pssm_output_dim)
        self.dropout = nn.Dropout(self.drop)

        # Combined layers
        self.fc = nn.Linear(self.graph_pssm_output_dim * 2, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, residue_features, residue_features2):
        # Process both proteins efficiently
        protein_feature = self.forward_part(residue_features)
        protein_feature2 = self.forward_part(residue_features2)
        # Removed randomness for consistent behavior
        feature = torch.cat((protein_feature, protein_feature2), dim=1)
        
        x = self.fc(feature)
        x = F.relu(x, inplace=True)  # Use inplace if possible
        x = self.dropout(x)
        out = self.out(x)
        output = torch.sigmoid(out)
        return output
    
    def forward_part(self, residue_features):
        batch_size = residue_features.shape[0]
        g_feature = F.relu(self.fc_g1(residue_features), inplace=True)
        g_feature = g_feature.view(batch_size, -1)  # Use view for efficiency
        g_feature = F.relu(self.fc_g2(g_feature), inplace=True)
        return g_feature
