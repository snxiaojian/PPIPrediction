import json
from collections import defaultdict
import sys
sys.path.append("./")
from ppi_network.static_args import *

class BioGridParser(object):
    def __init__(self, tab_file: str, map_file: str, taxon: str):
        self.taxon = taxon
        self.ppi_count = 0
        self.network = self.get_biogrid_network(tab_file, map_file)

    @staticmethod
    def parsed_network(species):
        tab2_file = "./data/biogrid/BIOGRID-ALL-4.4.214.tab2.txt"
        mapping_file = "./data/biogrid/UNIPROT.tab.txt"
        taxon = str(taxid_dict[species])
        parser = BioGridParser(tab2_file, mapping_file, taxon)
        return parser.network, parser.ppi_count

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
        existing_ppi = set()
        with open(path_to_network) as fp:
            for line in fp:
                if line.startswith('#'):
                    continue
                entries = line.strip().split('\t')
                biogrid_a, biogrid_b = entries[3], entries[4]
                organism_a, organism_b = entries[15], entries[16]
                if organism_a != self.taxon or organism_b != self.taxon:
                    continue
                try:  # if no matched accession found, pass it
                    protein_a = mapping[biogrid_a]
                    protein_b = mapping[biogrid_b]
                except KeyError:
                    continue
                # BioGRID interaction doesn't provide confidence score (mostly),
                # so we construct unweighted graph here
                if protein_a + protein_b in existing_ppi:
                    continue
                existing_ppi.add(protein_a + protein_b)
                existing_ppi.add(protein_b + protein_a)
                network[protein_a][protein_b] = 1
        self.ppi_count = len(existing_ppi)/2
        return network
