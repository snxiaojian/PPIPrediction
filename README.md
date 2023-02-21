### Envirment prepare
Please read envirment.txt
### Date prepare
#### Download species fasta file for training and test
sh ./data_process/fasta/collect_fasta.sh
#### Download custom database for BLAST
sh ./data_process/pssm/prepare_database.sh
#### Create custom database
python ./data_process/pssm/create_customdb.py
#### Generate PSSM file
python ./data_process/pssm/PSSMGenerator.py
#### Generate residue feature
python ./data_process/residue_feature_generate.py
#### Filter input fasta files
pthon ./data_process/filter_fasta.py
#### Generate dataset
python ./data_process/generate_dataset.py