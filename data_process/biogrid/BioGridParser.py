import json
from collections import defaultdict


class BioGridParser(object):
    def __init__(self, tab_file: str, map_file: str, taxon_set: set):
        self.taxon_set = taxon_set
        self.network = self.get_biogrid_network(tab_file, map_file)

    @staticmethod
    def parsed_network():
        tab2_file = "./data/biogrid/BIOGRID-ALL-4.4.214.tab2.txt"
        mapping_file = "./data/biogrid/UNIPROT.tab.txt"
        taxon_set = {'9606','7227', '3702', '559292'}
        return BioGridParser(tab2_file, mapping_file, taxon_set).network

    def biogrid2uniprot(self, file_path):
        mapping = dict()
        with open(file_path) as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                uniprot_id, biogrid_id, *_ = line.split()
                mapping[biogrid_id] = uniprot_id
        return mapping

    def get_biogrid_network(self, path_to_network, path_to_mapping):
        network = defaultdict(dict)
        mapping = self.biogrid2uniprot(path_to_mapping)
        with open(path_to_network) as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                entries = line.strip().split('\t')
                biogrid_a, biogrid_b = entries[3], entries[4]
                organism_a, organism_b = entries[15], entries[16]
                if not organism_a in self.taxon_set or not organism_b in self.taxon_set:
                    continue
                try:  # if no matched accession found, pass it
                    protein_a = mapping[biogrid_a]
                    protein_b = mapping[biogrid_b]
                except KeyError:
                    continue
                # BioGRID interaction doesn't provide confidence score (mostly),
                # so we construct unweighted graph here
                network[protein_a][protein_b] = network[protein_b][protein_a] = 1
        return network
