import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy
import csv
from train_util import *
sys.path.append("./")
from ppi_network.ReasonDatasetFastH5 import ReasonDatasetFastH5, fast_no_go_collate
from ppi_network.PSFastPPIModel import PSFastPPIModel
from ppi_network.static_args import *
from data_process.util import fasta_folder_from_feature_type, id_from_record

def reasoning(model, loader, ids):
    model.eval()
    total_preds = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for residue_with_pssm, residue_with_pssm2 in tepoch:
                output = model(toTensor(residue_with_pssm), toTensor(residue_with_pssm2))
                output = torch.round(output)
                index = total_preds.shape[0]
                write_result_to_local(output.cpu().numpy().flatten(), index, ids)
                total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)

def model_dir():
    return './data/PSFastPPIModels/'

def write_result_to_local(output, index, ids):
    indexes = numpy.where(output==1) + index
    pairsList = [pairs_for_index(i, ids) for i in indexes]        
    file = model_dir() + "reason_result" + "_ppi.tsv"
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(pairsList)
    
def reason(species):
    folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
    file = folder + species + ".fasta"
    ids = id_from_record(file)
    shuffle = False
    batch_size = 5000
    pick_num = pick_num_fast
    workers = 4
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
    
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    newest_model_path = find_newest_model(model_dir())
    if newest_model_path != None:
        checkpoint = torch.load(model_dir() + newest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    reasoning(model,test_loader, ids)


if __name__ == "__main__":
    reason("arabidopsis")