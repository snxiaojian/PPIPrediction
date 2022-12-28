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

cross_species_validation_species = ["C-elegans", "Ecoli-K12-W3110", "fission-yeast", "SARS-CoV-2"]

reason_species = ["NDH108"]

reasoning_species = ["peanut"]

dataset_for_training = "dataset_for_training"
dataset_for_reasoning = "dataset_for_reasoning"


taxid_dict={"Human": 9606,
            "fly":7227,
            "yeast":559292,
            "arabidopsis":3702,
            "C-elegans":6239,
            "Ecoli-K12-W3110":316407,
            "fission-yeast":284812,
            "SARS-CoV-2":2697049}