import torch
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling
import torch.nn.functional as F
import random

class GPSSPPIModel2(torch.nn.Module):

    def __init__(self, batch_size, device, pick_num):
        super(GPSSPPIModel2,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.pick_num = pick_num
        self.graph_embedding_size = 1024
        self.go_embedding_size = 1024 * 3
        self.pssm_embedding_size = pick_num * 20
        self.drop = 0.2

        # middle output size before combine
        self.middle_output_dim = 8
        
        # gcn
        self.gcn1 = GATConv(self.graph_embedding_size,self.graph_embedding_size,3)
        self.gcn2 = GATConv(self.graph_embedding_size*3,self.graph_embedding_size*3,3)
        self.gcn3 = GATConv(self.graph_embedding_size*9,self.graph_embedding_size*9,1)
        self.relu = torch.nn.ReLU()
        self.fc_graph1 = torch.nn.Linear((self.graph_embedding_size * 9), 400)
        self.fc_graph2 = torch.nn.Linear(400, self.middle_output_dim)
        
        self.fc_go_1 = torch.nn.Linear(self.go_embedding_size, 192)
        self.fc_go_2 = torch.nn.Linear(192, self.middle_output_dim)
        
        self.fc_pssm_1 = torch.nn.Linear(self.pssm_embedding_size, 125)
        self.fc_pssm_2 = torch.nn.Linear(125, self.middle_output_dim)

        self.maxpooling = MaxPooling()
        self.dropout = torch.nn.Dropout(self.drop)


        # combined layers
        self.fc1 = torch.nn.Linear(self.middle_output_dim * 3 * 2, self.middle_output_dim)
        self.out = torch.nn.Linear(self.middle_output_dim, 1)

    # input1 input2
    def forward(self, G_residue, go_embedding, pssm, indexes, G_residue2, go_embedding2, pssm2, indexes2):
        graph_feature, go_feature, pssm = self.forward_part(G_residue, go_embedding, pssm)
        graph_feature2, go_feature2, pssm2 = self.forward_part(G_residue2, go_embedding2, pssm2)
        if bool(random.getrandbits(1)):
            feature = torch.cat((graph_feature2, go_feature2, pssm2, graph_feature, go_feature, pssm), dim=1)
        else:
            feature = torch.cat((graph_feature, go_feature, pssm, graph_feature2, go_feature2, pssm2), dim=1)
        
        x = self.fc1(feature)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        output = torch.sigmoid(out)
        return output
    
    def forward_part(self, G_residue, go_embedding, pssm):
        # graph
        graph_feature = self.relu(self.gcn1(G_residue,G_residue.ndata['feat']))
        graph_feature = graph_feature.reshape(-1,self.graph_embedding_size*3)
        graph_feature = self.relu(self.gcn2(G_residue, graph_feature))
        graph_feature = graph_feature.reshape(-1,self.graph_embedding_size*9)
        graph_feature = self.relu(self.gcn3(G_residue, graph_feature))
        graph_feature = graph_feature.reshape(-1,self.graph_embedding_size*9)

        g1_maxpooling = self.maxpooling(G_residue,graph_feature)  

        
        graph_feature = self.relu(self.fc_graph1(g1_maxpooling))
        graph_feature = self.relu(self.fc_graph2(graph_feature))

        go_embedding = self.relu(self.fc_go_1(go_embedding))
        go_embedding = self.relu(self.fc_go_2(go_embedding))

        pssm = self.relu(self.fc_pssm_1(pssm.reshape(-1, self.pssm_embedding_size)))
        pssm = self.relu(self.fc_pssm_2(pssm))
        
        return graph_feature, go_embedding, pssm