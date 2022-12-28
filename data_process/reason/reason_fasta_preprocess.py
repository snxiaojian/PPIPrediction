from Bio import SeqIO
import sys
sys.path.append("./")
from data_process.pssm.PSSMGenerator import PSSMGenerator
from data_process.filter_fasta.filter_fasta import start_filter
from ppi_network.static_args import *
from data_process.util import fasta_folder_from_feature_type, id_from_record
from data_process.tensor_to_disk.save_dataset_tensor_to_disk import write_records_to_disk

def generate_pssm(file, species):
    pssm_generator = PSSMGenerator(file, species)
    pssm_generator.generate()
    
if __name__ == "__main__":
    file = "./data/input/NDH108.fasta"
    species = "NDH108"
    # generate_pssm(file, species)
    start_filter(feature_type_residue_pssm)
    # folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
    # file = folder + species + ".fasta"
    # recordIDs = {}
    # for record in SeqIO.parse(file, "fasta"):
    #     recordIDs[id_from_record(record)] = record
    # write_records_to_disk(recordIDs, feature_type_residue_pssm)