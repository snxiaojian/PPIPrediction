from go_rgcn.OBOParser import GODag

class GeneOntology(dict):
    def __init__(self, obo_file_path):
        super(GeneOntology, self).__init__()
        go_dag = GODag(obo_file_path, ['relationship','intersection_of'])
        self.alt_ids = go_dag.alt_ids
        for go_id, go_term in go_dag.items():
            self[go_id] = go_term
