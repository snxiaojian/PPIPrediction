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
from ppi_network.static_args import *

class ProteinFeatureLoader:
    def __init__(self, feature_type):
        self.feature_type = feature_type
        self.anotations = GOAParser.parsed_annotation()
        self.gene_ontology = self.gene_ontology()
        self.records, self.species_dict = records_from_filtered_input(feature_type)
        self.residue_feature_loader = ResidueFeatureLoader()
        self.all_embedding_dict = self.load_embedding_dict()
    
    def load_embedding_dict(self):
        embedding_dict = numpy.load("./data/go/go_embedding.npy", allow_pickle=True)
        for idx, j in numpy.ndenumerate(embedding_dict):
            return j
    def gene_ontology(self):
        obo_file = "./data/go/go.obo"
        return GeneOntology(obo_file_path=obo_file)
    
    def go_embedding_zero_if_empty(self,go_embedding):
        if len(go_embedding) == 0:
            go_embedding = [[0] * go_embedding_size]
        return torch.sum(torch.Tensor(go_embedding), dim=0)
        
    def default_loader(self, pid):
        record = self.records[pid]
        gene = gene_from_record(record)
        species = self.species_dict[pid]
        
        if record is None:
            raise ValueError("No record for %s" % pid)
        
        if self.feature_type == feature_type_go_graph_pssm:
        # go embedding
            if pid in self.anotations:
                go_ids = self.anotations[pid]
            if gene in self.anotations:
                go_ids = self.anotations[gene]
            if go_ids is None:
                raise ValueError("No anotation for %s" % pid)
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
        
        pssm_file = "./data/pssm/" + species + "/" + pid + ".pssm"
        if not os.path.exists(pssm_file):
            raise ValueError("No pssm for %s" % pid)
        
        lenth_of_protein = len(record.seq._data)
        # residue graph
        all_residue_features = self.residue_feature_loader.load_residue_feature(record)
        all_residue_features = torch.Tensor(all_residue_features)
        
        
        if self.feature_type != feature_type_residue_pssm:
            contact_map_file = "./data/contact_map/" + species + "/" + pid + ".npz"
            contact_map = numpy.load(contact_map_file)["contact"]
            if contact_map is None:
                raise ValueError("No contact map for %s" % pid)
            

            if not os.path.exists(contact_map_file):
                raise ValueError("No contact map for %s" % pid)
            adj_residue = spp.coo_matrix(contact_map)
            G_residue = dgl.from_scipy(adj_residue)
            G_residue.ndata['feat'] = all_residue_features

        # pssm feature
        pssm = PSSMGenerator.readFromPSSM(pssm_file)
        pssm = torch.Tensor(pssm)
        pssm = torch.nn.functional.normalize(pssm, dim=1)
        
        # pick  residues
        indexes = ResiduePicker(record=record, pssm=pssm).pick_residue(number=pick_num_precise)
        
        selected_residue_features = []
        
        picked_pssm = []
        for index in indexes:
            if index < 0:
                picked_pssm.append([0] * 20)
                selected_residue_features.append([0] * (20 + residue_embedding_size))
            elif index < lenth_of_protein:
                picked_pssm.append(pssm[index].tolist())
                residue_feature = pssm[index].tolist() + all_residue_features[index].tolist()
                selected_residue_features.append(residue_feature)
            else:
                raise ValueError("index error")
        picked_pssm = torch.Tensor(picked_pssm)
        
        # last index add length of protein
        indexes.append(lenth_of_protein)
        indexes = torch.IntTensor(indexes)
        selected_residue_features = torch.Tensor(selected_residue_features)
        if self.feature_type == feature_type_go_graph_pssm:
            return G_residue, go_embedding, picked_pssm, indexes
        elif self.feature_type == feature_type_residue_pssm:
            return selected_residue_features[0:pick_num_fast]
        elif self.feature_type == feature_type_graph_pssm:
            return G_residue, picked_pssm, indexes
        else:
            raise ValueError("feature type error")

if __name__ == "__main__":
    ProteinFeatureLoader(feature_type_residue_pssm).default_loader("A9YTQ3")