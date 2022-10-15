import torch
from torch_geometric.data import Data

from GeneOntology import *
from OBOParser import *


class GOPYGConverter(object):
    def __init__(self) -> None:
        obo_file = "./data/go/go.obo"
        self.ontology = GeneOntology(obo_file_path=obo_file)

    def go_id2pyg_data(self, go_id: str):
        go_term = self.ontology[go_id]
        first_level_relations = self.relations_for_term(go_term=go_term)
        second_level_relations = []
        for relation in first_level_relations:
            second_level_relations.extend(
                self.relations_for_term(go_term=self.ontology[relation[2]]))
        relations = first_level_relations + second_level_relations
        node_sets = set()
        node_value_list = []
        for relation in relations:
            node_sets.add(relation[0])
            node_sets.add(relation[2])
        for node in list(node_sets):
            index = self.ontology[node].index
            node_value = GOPYGConverter.generate_one_hot(
                num=index, dim=len(self.ontology))
            node_value_list.append(node_value)
        edge_index, edge_type = self.build_adj(
            relations=relations, nodes=list(node_sets))
        pyg_data = Data(
            x=torch.tensor(node_value_list, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
            edge_type=torch.tensor(edge_type, dtype=torch.long)
        )
        pyg_data.is_directed = True
        return pyg_data

    def build_adj(self, relations: list, nodes: list):
        edge_index, edge_type = [], []
        for relation in relations:
            relation0_index = nodes.index(relation[0])
            relation2_index = nodes.index(relation[2])
            # direction is from relation2 to relation0
            edge_index.append([relation2_index, relation0_index])
            edge_type.append(relation[1])
        return edge_index, edge_type

    def relations_for_term(self, go_term: GOTerm):
        relations = []
        for parent in go_term.parents:
            type_value = self.type_value_for_relation("is_a")
            relations.append([go_term.id, type_value, parent])
        if hasattr(go_term, "intersection_of"):
            for intersection_of, terms in go_term.intersection_of.items():
                type_value = self.type_value_for_relation(intersection_of)
                for term in terms:
                    relations.append([go_term.id, type_value, term])
        if hasattr(go_term, "relationship"):
            for relationship, terms in go_term.relationship.items():
                type_value = self.type_value_for_relation(relationship)
                for term in terms:
                    relations.append([go_term.id, type_value, term])
        return relations

    all_relations: list[str] = [
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
    relations = all_relations[1:6]
    intersections = all_relations[6:]

    def type_value_for_relation(self, relation: str):
        return GOPYGConverter.all_relations.index(relation)

    @staticmethod
    def generate_one_hot(num, dim):
        vec = [0] * dim
        vec[num] = 1
        return vec


if __name__ == "__main__":
    converter = GOPYGConverter()
    data = converter.go_id2pyg_data("GO:0000018")
