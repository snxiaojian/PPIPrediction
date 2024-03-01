import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy
import csv
from train_util import *
sys.path.append("./")
from ppi_network.ReasonDatasetFastH5 import ReasonDatasetFastH5, pid_pairs_for
from ppi_network.PSFastPPIModel import PSFastPPIModel
from ppi_network.static_args import *
from data_process.util import *
import time

def reasoning(model, loader, ids, species, startIndex):
    model.eval()
    total_preds = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    file_name = model_dir() + "reason_result_" + species + "_ppi.tsv"
    if os.path.exists(file_name):
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        os.rename(file_name, file_name + timestamp)
    predicted_number = startIndex
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for residue_with_pssm, residue_with_pssm2 in tepoch:
                output = model(residue_with_pssm.to(device), residue_with_pssm2.to(device))
                write_result_to_local(output.cpu().numpy().flatten(), predicted_number, ids, file_name, tepoch.n)
                predicted_number = predicted_number + output.shape[0]
                total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)

def model_dir():
    return './data/PSFastPPIModels/'

def write_result_to_local(output, predicted_number, ids, file_name, batch):
    indexes = numpy.where(output>0.5)[0]
    pairsList = [pid_pairs_for(ids, i + predicted_number) + (output[i],) for i in indexes]
    pairsList.insert(0, ["batch", "start", batch])
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(pairsList)
    
def reason(species):
    folder = fasta_folder_from_feature_type(feature_type_residue_pssm)
    file = folder + species + ".fasta"
    ids = ids_from_fasta_file(file)
    shuffle = False
    batch_size = 20000
    pick_num = pick_num_fast
    workers = 0
    drop_last = False
    pin_memory = False
    
    startIndex = (252484 + 67168)*batch_size
    test_dataset = ReasonDatasetFastH5(species, startIndex)
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               num_workers=workers,
               pin_memory=pin_memory)

    model_path = model_dir() + 'epoch' + "62" + '.pkl'
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    reasoning(model,test_loader, ids, species, startIndex)


if __name__ == "__main__":
    reason("arabidopsis")