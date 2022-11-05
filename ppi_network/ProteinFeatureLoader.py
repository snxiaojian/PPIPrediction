import sys
import os
import dgl
import scipy.sparse as spp
import numpy
import torch

sys.path.append("./")
from go_rgcn.GOAParser import GOAParser
from ppi_network.ResidueFeatureLoader import ResidueFeatureLoader
from data_process.util import records_from_filtered_input, gene_from_record
from data_process.pssm.PSSMGenerator import PSSMGenerator

device = torch.device('cuda')
class ProteinFeatureLoader:
    def __init__(self):
        self.anotations = GOAParser.parsed_annotation()
        self.records, self.species_dict = records_from_filtered_input()
        self.residue_feature_loader = ResidueFeatureLoader()
        self.all_embedding_dict = self.load_embedding_dict()
    
    def load_embedding_dict(self):
        embedding_dict = numpy.load("./data/go/go_embeddings.npy", allow_pickle=True)
        for idx, j in numpy.ndenumerate(embedding_dict):
            return j
    
    def default_loader(self, pid):
        record = self.records[pid]
        gene = gene_from_record(record)
        species = self.species_dict[pid]
        if record is None:
            raise ValueError("No record for %s" % pid)
        if pid in self.anotations:
            go_ids = self.anotations[pid]
        if gene in self.anotations:
            go_ids = self.anotations[gene]
        if go_ids is None:
            raise ValueError("No anotation for %s" % pid)
        
        pssm_file = "./data/pssm/" + species + "/" + pid + ".pssm"
        if not os.path.exists(pssm_file):
            raise ValueError("No pssm for %s" % pid)
        
        contact_map_file = "./data/contact_map/" + species + "/" + pid + ".npz"
        contact_map = numpy.load(contact_map_file)["contact"]
        if contact_map is None:
            raise ValueError("No contact map for %s" % pid)
        

        if not os.path.exists(contact_map_file):
            raise ValueError("No contact map for %s" % pid)
        # residue graph
        residue_feature = self.residue_feature_loader.load_residue_feature(record)
        residue_feature = torch.Tensor(residue_feature).to(device)
        adj_residue = spp.coo_matrix(contact_map)
        G_residue = dgl.from_scipy(adj_residue, device=device)
        G_residue.ndata['feat'] = residue_feature
        # go embedding
        embedding_dict = numpy.load("./data/go/go_embeddings.npy", allow_pickle=True)
        for idx, j in numpy.ndenumerate(embedding_dict):
            embedding_dict = j
        go_embedding = []
        for go_id in go_ids:
            if go_id in embedding_dict:
                e = embedding_dict[go_id]
                go_embedding.append(e)
        go_embedding = torch.Tensor(go_embedding).to(device)
        go_embedding = torch.sum(go_embedding, dim=0)
        # pssm feature
        pssm = PSSMGenerator.readFromPSSM(pssm_file)
        pssm = torch.from_numpy(pssm).to(device)
        return G_residue, go_embedding, pssm

if __name__ == "__main__":
    ProteinFeatureLoader.default_loader("A0A0A7EPL0")