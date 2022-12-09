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
from ppi_network.ResiduePicker import ResiduePicker
from go_rgcn.GeneOntology import GeneOntology


class ProteinFeatureLoader:
    def __init__(self, pick_num):
        self.go_embedding_size = 1024
        self.anotations = GOAParser.parsed_annotation()
        self.gene_ontology = self.gene_ontology()
        self.records, self.species_dict = records_from_filtered_input()
        self.residue_feature_loader = ResidueFeatureLoader()
        self.all_embedding_dict = self.load_embedding_dict()
        self.pick_number = pick_num
    
    def load_embedding_dict(self):
        embedding_dict = numpy.load("./data/go/go_embedding.npy", allow_pickle=True)
        for idx, j in numpy.ndenumerate(embedding_dict):
            return j
    def gene_ontology(self):
        obo_file = "./data/go/go.obo"
        return GeneOntology(obo_file_path=obo_file)
    
    def go_embedding_zero_if_empty(self,go_embedding):
        if len(go_embedding) == 0:
            go_embedding = [[0] * self.go_embedding_size]
        return torch.sum(torch.Tensor(go_embedding), dim=0)
        
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
        
        lenth_of_protein = len(record.seq._data)
        
        # residue graph
        residue_feature = self.residue_feature_loader.load_residue_feature(record)
        residue_feature = torch.Tensor(residue_feature)
        adj_residue = spp.coo_matrix(contact_map)
        G_residue = dgl.from_scipy(adj_residue)
        G_residue.ndata['feat'] = residue_feature
        # go embedding
        go_embedding_bp = []
        go_embedding_mf = []
        go_embedding_cc = []
        for go_id in go_ids:
            if go_id in self.all_embedding_dict:
                e = self.all_embedding_dict[go_id].tolist()
                go_term = self.gene_ontology[go_id]
                if go_term.namespace == "biological_process":
                    go_embedding_bp.append(e)
                if go_term.namespace == "molecular_function":
                    go_embedding_mf.append(e)
                if go_term.namespace == "cellular_component":
                    go_embedding_cc.append(e)
        go_embedding_bp = self.go_embedding_zero_if_empty(go_embedding_bp)
        go_embedding_mf = self.go_embedding_zero_if_empty(go_embedding_mf)
        go_embedding_cc = self.go_embedding_zero_if_empty(go_embedding_cc)
        go_embedding = torch.cat((go_embedding_bp, go_embedding_mf, go_embedding_cc), dim = 0)
        # pssm feature
        pssm = PSSMGenerator.readFromPSSM(pssm_file)
        
        
        # pick  residues
        indexes = ResiduePicker(record=record, pssm=pssm).pick_residue(number=self.pick_number)
        
        picked_pssm = []
        for index in indexes:
            if index < 0:
                picked_pssm.append([0] * 20)
            elif index < lenth_of_protein:
                picked_pssm.append(pssm[index])
            else:
                raise ValueError("index error")
        picked_pssm = torch.Tensor(picked_pssm)
        norm_picked_pssm = torch.nn.functional.normalize(picked_pssm, dim=1)
        
        # last index add length of protein
        indexes.append(lenth_of_protein)
        indexes = torch.IntTensor(indexes)
        return G_residue, go_embedding, norm_picked_pssm, indexes

if __name__ == "__main__":
    ProteinFeatureLoader(pick_num= 50).default_loader("A9YTQ3")