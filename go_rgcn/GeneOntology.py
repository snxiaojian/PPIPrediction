"""Definition of Gene Ontology.
"""
from collections import defaultdict
from OBOParser import GODag


def get_short_ns(long_ns):
    """Return short namespace id of the given long namespace.
    :param long_ns: long name of namespace
    :return: short namespace id
    """
    if long_ns == "biological_process":
        return "bp"
    elif long_ns == "cellular_component":
        return "cc"
    elif long_ns == "molecular_function":
        return "mf"
    else:
        raise ValueError("Can't recognize %s" % long_ns)


class GeneOntology(dict):
    """Definition of GO, it is inheritance of dict.
    Attributes:
        - alt_ids: dict, alternative ids of GO terms, like
            { go_term1: [ alt_id1, alt_id2, ... ], ... }
        - dict of GO terms, you can visit it like dict,
            e.g. ontology['GO:0000005']
    """
    def __init__(self, obo_file_path):
        """
        :param obo_file_path: path to obo file
        :return: None
        """
        super(GeneOntology, self).__init__()
        go_dag = GODag(obo_file_path, ['relationship','intersection_of'])
        self.alt_ids = go_dag.alt_ids
        for go_id, go_term in go_dag.items():
            self[go_id] = go_term
