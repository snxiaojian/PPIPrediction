import sys
import os
import dgl
import scipy.sparse as spp
import numpy
import torch
from ResidueFeatureLoader import ResidueFeatureLoader
sys.path.append("./")
from go_rgcn.GOAParser import GOAParser
from go_rgcn.GOGraphInfoProvider import GOGraphInfoProvider
from data_process.util import records_from_filtered_input, gene_from_record
from data_process.PSSM.PSSM_generator import PSSMGenerator

device = torch.device('cuda')
class ProteinFeatureLoader:
    anotations = GOAParser.parsed_annotation()
    records, species_dict = records_from_filtered_input()
    residue_feature_loader = ResidueFeatureLoader()
    go_graph_info_provider = GOGraphInfoProvider()
    go_graph = dgl.heterograph(go_graph_info_provider.graph_dict)
    
    @staticmethod
    def default_loader(pid):
        record = ProteinFeatureLoader.records[pid]
        gene = gene_from_record(record)
        species = ProteinFeatureLoader.species_dict[pid]
        if record is None:
            raise ValueError("No record for %s" % pid)
        if pid in ProteinFeatureLoader.anotations:
            go_id = ProteinFeatureLoader.anotations[pid]
        if gene in ProteinFeatureLoader.anotations:
            go_id = ProteinFeatureLoader.anotations[gene]
        if go_id is None:
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
        residue_feature = ProteinFeatureLoader.residue_feature_loader.load_residue_feature(record)
        residue_feature = torch.Tensor(residue_feature).to(device)
        adj_residue = spp.coo_matrix(contact_map)
        G_residue = dgl.from_scipy(adj_residue, device=device)
        G_residue.ndata['feat'] = residue_feature
        # go graph
        go_indexes = ProteinFeatureLoader.go_graph_info_provider.layer_related_indexes_for_ids(go_id)

        go_indexes = torch.tensor(go_indexes, dtype=torch.int64)
        G_go = ProteinFeatureLoader.go_graph.subgraph(go_indexes, relabel_nodes=True)
        G_go = G_go.to(device)
        pssm_feature = PSSMGenerator.readFromPSSM(pssm_file)
        pssm_feature = torch.from_numpy(pssm_feature).to(device)
        return G_residue, G_go, pssm_feature
        

if __name__ == "__main__":
    ProteinFeatureLoader.default_loader("A0A0A7EPL0")