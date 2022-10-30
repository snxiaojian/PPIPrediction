import sys
import torch
sys.path.append("./")
from go_rgcn.GeneOntology import GeneOntology
from go_rgcn.OBOParser import *

class GOGraphInfoProvider(object):
    def __init__(self) -> None:
        obo_file = "./data/go/go.obo"
        self.ontology = GeneOntology(obo_file_path=obo_file)
        self.go_ids = list(self.ontology.keys())
        self.triplets_relations = self._triplets_relations()
        self.graph_dict = self._graph_dict_for_all()
        self._all_onehot_features = self._onehot_features_for_all()
    
    def _layer_related_ids_for_ids(self, ids: list):
        layer_related_ids = set()
        for go_id in ids:
            go_term = self.ontology[go_id]
            first_layer_relations = self.relations_for_term(go_term=go_term)
            for relation in first_layer_relations:
                layer_related_ids.add(relation[2])
                second_layer_go_term = self.ontology[relation[2]]
                relations = self.relations_for_term(go_term=second_layer_go_term)
                for r in relations:
                    layer_related_ids.add(r[2])
        return list(layer_related_ids)
    
    def layer_related_indexes_for_ids(self, ids: list):
        ids = self._layer_related_ids_for_ids(ids)
        indexes = [self.go_ids.index(go_id) for go_id in ids]
        return indexes
    
    def _onehot_features_for_all(self):
        onehot_features = []
        dim = len(self.go_ids)
        for index, go_id in enumerate(self.go_ids):
            onehot_feature = GOGraphInfoProvider.generate_one_hot(num=index, dim=dim)
            onehot_features.append(onehot_feature)
        return onehot_features
    
    def onehot_features_for_indexes(self, indexes: list):
        features = []
        for index in indexes:
            features.append(self._all_onehot_features[index])
        return features
    
    def _graph_dict_for_all(self):
        graph_dict = {}
        source_nodes = defaultdict(list)
        destination_nodes = defaultdict(list)
        for relation in self.triplets_relations:
            relation_type = relation[1]
            for i in range(0, len(GOGraphInfoProvider.all_relation_types)):
                if relation_type == GOGraphInfoProvider.all_relation_types[i]:
                    key = ("go_id", relation_type, "go_id")
                    # 2 provide infomation to 0
                    source_index = self.go_ids.index(relation[2])
                    source_nodes[key].append(source_index)
                    destination_index = self.go_ids.index(relation[0])
                    destination_nodes[key].append(destination_index)
        for key in source_nodes.keys(): 
            source = torch.tensor(source_nodes[key])
            destination = torch.tensor(destination_nodes[key])
            graph_dict[key] = (source, destination)
        return graph_dict
    
    def _triplets_relations(self):
        all_triplets_relations = []
        for go_id in self.go_ids:
            go_term = self.ontology[go_id]
            first_layer_relations = self.relations_for_term(go_term=go_term)
            all_triplets_relations.extend(first_layer_relations)  
        return all_triplets_relations            

    def relations_for_term(self, go_term: GOTerm):
        relations = []
        for parent in go_term.parents:
            relations.append([go_term.id, "is_a", parent])
        if hasattr(go_term, "intersection_of"):
            for intersection_of, terms in go_term.intersection_of.items():
                for term in terms:
                    relations.append([go_term.id, intersection_of, term])
        if hasattr(go_term, "relationship"):
            for relationship, terms in go_term.relationship.items():
                for term in terms:
                    relations.append([go_term.id, relationship, term])
        return relations

    all_relation_types: list[str] = [
        "is_a",
        "relationship has_part",
        "relationship part_of",
        "relationship regulates",
        "relationship positively_regulates",
        "relationship negatively_regulates",
        "intersection_of",
        "intersection_of regulates",
        "intersection_of positively_regulates",
        "intersection_of negatively_regulates",
        "intersection_of part_of",
    ]

    @staticmethod
    def generate_one_hot(num, dim):
        vec = [0] * dim
        vec[num] = 1
        return vec
    
if __name__ == "__main__":
    go_graph_info_provider = GOGraphInfoProvider()
    dict = go_graph_info_provider.graph_dict
    indexes = go_graph_info_provider.layer_related_indexes_for_ids(["GO:0008153"])
    feat = go_graph_info_provider.onehot_features_for_indexes(indexes)
    print(feat)