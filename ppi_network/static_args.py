# in fast model, select 50 animo acid
pick_num_fast = 50
# in precise model, select 100 animo acid
pick_num_precise = 100

pssm_size = 20

go_embedding_size = 1024
residue_embedding_size = 1024

feature_type_go_graph_pssm = "feature_type_go_graph_pssm"
feature_type_graph_pssm = "feature_type__graph_pssm"
feature_type_residue_pssm = "feature_type_residue_pssm"

train_species = ["arabidopsis", "fly", "human", "yeast"]

test_other_species = ["C-elegans", "mouse", "fission-yeast"]

reason_species = ["NDH108"]

taxid_dict={"human": 9606,
            "fly":7227,
            "yeast":559292,
            "arabidopsis":3702,
            "C-elegans":6239,
            "fission-yeast":284812,
            "mouse":10090
            }