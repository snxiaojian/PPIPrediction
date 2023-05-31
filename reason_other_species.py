import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy
import csv
from train_util import *
sys.path.append("./")
from ppi_network.ReasonDatasetFastH5 import ReasonDatasetFastH5, fast_no_go_collate, pid_pairs_for
from ppi_network.PSFastPPIModel import PSFastPPIModel
from ppi_network.static_args import *
from data_process.util import *

def reasoning(model, loader, ids, species):
    model.eval()
    total_preds = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    file_name = model_dir() + "reason_result_" + species + "_ppi.tsv"
    if os.path.exists(file_name):
        os.remove(file_name)
    predicted_number = 0
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for residue_with_pssm, residue_with_pssm2 in tepoch:
                output = model(toTensor(residue_with_pssm), toTensor(residue_with_pssm2))
                write_result_to_local(output.cpu().numpy().flatten(), predicted_number, ids, file_name)
                predicted_number = predicted_number + output.shape[0]
                total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)

def model_dir():
    return './data/PSFastPPIModels/'

def write_result_to_local(output, predicted_number, ids, file_name):
    indexes = numpy.where(output>0.5)[0]
    pairsList = [pid_pairs_for(ids, i + predicted_number) + (output[i],) for i in indexes]        
    # append new content to file
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(pairsList)
    
def reason(species):
    folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
    file = folder + species + ".fasta"
    ids = ids_from_fasta_file(file)
    shuffle = False
    batch_size = 10000
    pick_num = pick_num_fast
    workers = 2
    drop_last = False
    pin_memory = True
    
    test_dataset = ReasonDatasetFastH5(species)
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)

    model_path = model_dir() + 'epoch' + "62" + '.pkl'
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    reasoning(model,test_loader, ids, species)


if __name__ == "__main__":
    reason("arabidopsis")