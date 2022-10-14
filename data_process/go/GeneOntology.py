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



class GOTerm(object):
    def __init__(self, go_term):
        """Definition of an GO term.
        Attributes:
            - id: accession of GO term, e.g. GO:0000005
            - parents: set, parents of this term
            - name: name of GO term
            - ns: namespace the term belongs to
            - children: set, children of this term
            - depth: the depth of term in the whole GO
        :param go_term: instance of the HPO term
        :return: None
        """
        self.id = go_term.id
        self.parents = go_term.parents
        self.name = go_term.name
        self.ns = get_short_ns(go_term.namespace)


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
            self[go_id] = GOTerm(go_term)
