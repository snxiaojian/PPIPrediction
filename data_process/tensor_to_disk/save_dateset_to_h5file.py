import h5py
import os
from Bio import SeqIO
import sys
sys.path.append("./")
from data_process.util import records_from_filtered_input
from ppi_network.ProteinFeatureLoader import ProteinFeatureLoader
from ppi_network.static_args import *

h5file_folder = "./tensor/h5file/"
h5file =  h5file_folder + feature_type_residue_pssm + ".h5"

def key(pid,type,num_pick):
    return "pid:"+pid+"type:"+type+"picknum:"+str(num_pick)

def process(pid, index, loader, group):
    residue_with_pssm = loader(pid)
    keystr = key(pid,feature_type_residue_pssm,pick_num_fast)
    
    group.create_dataset(keystr, data=residue_with_pssm.tolist())
    if index%100 == 0:
        print("processed %d" % index)
        
def write_records_to_disk(recordIDs, species_dict):
    loader = ProteinFeatureLoader(feature_type_residue_pssm).default_loader
    print("total number of records: %d" % len(recordIDs))

    with h5py.File(h5file, 'w') as f:
        for species in set(species_dict.values()):
            f.create_group(species)
        for index, pid in enumerate(recordIDs):
            species = species_dict[pid]
            group = f[species]
            process(pid, index, loader, group)
    
def write_dataset_to_h5file(): 
    if not os.path.exists(h5file_folder):
        os.mkdir(h5file_folder)       
    recordIDs, species_dict = records_from_filtered_input(feature_type_residue_pssm)
    write_records_to_disk(recordIDs, species_dict)
    
def test_read():
    with h5py.File(h5file, 'r') as f:
        for species in f.keys():
            dset = f[species]
            keys =  list(dset.keys())
            for key in keys:
                value = dset[key][:]
                print(value)
if __name__ == "__main__":
    write_dataset_to_h5file()
    # test_read()