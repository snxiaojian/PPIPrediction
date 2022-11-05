import torch
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling
import torch.nn.functional as F

class GPSSPPIModel(torch.nn.Module):

    def __init__(self):
        super(GPSSPPIModel,self).__init__()
        # torch.backends.cudnn.enabled = False
        self.batch_size = 8
        self.g_embedding_size = 256
        self.drop = 0.2
        self.g_output_dim = 256
        self.go_output_dim = 256
        # gcn
        self.gcn1 = GATConv(self.g_embedding_size,self.g_embedding_size,3)
        self.gcn2 = GATConv(self.g_embedding_size*3,self.g_embedding_size*3,3)
        self.gcn3 = GATConv(self.g_embedding_size*9,self.g_embedding_size*9,1)
        self.relu = torch.nn.ReLU()
        self.fc_g = torch.nn.Linear(self.g_embedding_size*9, self.g_output_dim)

        self.maxpooling = MaxPooling()
        self.dropout = torch.nn.Dropout(self.drop)


        # combined layers
        self.fc1 = torch.nn.Linear((self.go_output_dim + self.g_output_dim)*2, 256)
        self.fc2 = torch.nn.Linear(256,32)
        self.out = torch.nn.Linear(32, 2)

    # input1 input2
    def forward(self,G_residue,go_embedding,pssm, G_residue2,go_embedding2,pssm2):
        # protein1
        g_feature, go_feature = self.forward_part(G_residue,go_embedding,pssm)
        g_feature2, go_feature2 = self.forward_part(G_residue2,go_embedding2,pssm2)
        feature = torch.cat((g_feature,go_feature,g_feature2,go_feature2),dim=1)
        
        x = self.fc1(feature)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        output = F.sigmoid(out)
        return output
    
    def forward_part(self,G_residue,go_embedding,pssm):
        g_feature = self.relu(self.gcn1(G_residue,G_residue.ndata['feat']))
        g_feature = g_feature.reshape(-1,self.embedding_size*3)
        g_feature = self.relu(self.gcn2(G_residue, g_feature))
        g_feature = g_feature.reshape(-1,self.embedding_size*9)
        g_feature = self.relu(self.gcn3(G_residue, g_feature))
        g_feature = g_feature.reshape(-1,self.embedding_size*9)
        g1_maxpooling = self.maxpooling(G_residue, g_feature)  
        # flatten
        g_feature = self.relu(self.fc_g(g1_maxpooling))

        return g_feature, go_embedding