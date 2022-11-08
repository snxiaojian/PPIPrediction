import torch
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling
import torch.nn.functional as F
import random

class GPSSPPIModel(torch.nn.Module):

    def __init__(self):
        super(GPSSPPIModel,self).__init__()
        torch.backends.cudnn.enabled = False
        self.pick_num = 50
        self.g_embedding_size = 256
        self.drop = 0.2
        # final residue size before combine acid
        self.residue_out_dim = 64
        # mix graph and pssm feature
        self.graph_pssm_output_dim = 64
        self.go_output_dim = 64
        self.pssm_size = 20
        self.pssm_out_dim = 64
        # gcn
        self.gcn1 = GATConv(self.g_embedding_size,self.g_embedding_size, 1)
        self.gcn2 = GATConv(self.g_embedding_size,self.g_embedding_size, 1)
        self.gcn3 = GATConv(self.g_embedding_size,self.g_embedding_size, 1)
        self.relu = torch.nn.ReLU()
        self.fc_g1 = torch.nn.Linear((self.g_embedding_size + self.pssm_size), self.residue_out_dim)
        self.fc_g2 = torch.nn.Linear(self.residue_out_dim * self.pick_num, 400)
        self.fc_g3 = torch.nn.Linear(400, self.graph_pssm_output_dim)
        
        self.fc_go_embedding = torch.nn.Linear(512, 64)

        self.maxpooling = MaxPooling()
        self.dropout = torch.nn.Dropout(self.drop)


        # combined layers
        self.fc1 = torch.nn.Linear((self.go_output_dim + self.graph_pssm_output_dim)*2, 256)
        self.fc2 = torch.nn.Linear(256,32)
        self.out = torch.nn.Linear(32, 1)

    # input1 input2
    def forward(self, G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2):
        # protein1
        graph_feature, go_feature = self.forward_part(G_residue, go_embedding, indexes, pssm)
        graph_feature2, go_feature2 = self.forward_part(G_residue2, go_embedding2, indexes2, pssm2)
        if bool(random.getrandbits(1)):
            feature = torch.cat((graph_feature2, go_feature2, graph_feature, go_feature), dim=1)
        else:
            feature = torch.cat((graph_feature, go_feature, graph_feature2, go_feature2), dim=1)
        
        x = self.fc1(feature)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        output = torch.sigmoid(out)
        return output
    
    def forward_part(self, G_residue, go_embedding, indexes, pssm):
        device = indexes.device
        batch_size = indexes.shape[0]
        split = torch.split(indexes, [self.pick_num,1], dim=1)
        indexes = split[0]
        protein_length = torch.squeeze(split[1])
        indexes_in_batch =  torch.cumsum(protein_length, dim=0)
        indexes_in_batch = torch.roll(indexes_in_batch, shifts=1, dims=0)
        indexes_in_batch[0] = 0
        indexes_in_batch = indexes_in_batch.unsqueeze(1)
        indexes_in_batch = torch.add(indexes, indexes_in_batch).reshape(-1).type(torch.IntTensor).to(device)
        
        g_feature = self.relu(self.gcn1(G_residue,G_residue.ndata['feat']))
        g_feature = g_feature.reshape(-1,self.g_embedding_size)
        g_feature = self.relu(self.gcn2(G_residue, g_feature))
        g_feature = g_feature.reshape(-1,self.g_embedding_size)
        g_feature = self.relu(self.gcn3(G_residue, g_feature))
        g_feature = g_feature.reshape(-1,self.g_embedding_size)
        selected_feature = torch.index_select(g_feature, 0, indexes_in_batch)
        
        residue_feature = torch.cat((selected_feature, pssm.reshape(-1, self.pssm_size)), dim=1)
        
        g_feature = self.relu(self.fc_g1(residue_feature))
        g_feature = g_feature.reshape(batch_size, -1)
        g_feature = self.relu(self.fc_g2(g_feature))
        g_feature = self.relu(self.fc_g3(g_feature))

        go_embedding = self.relu(self.fc_go_embedding(go_embedding))

        return g_feature, go_embedding