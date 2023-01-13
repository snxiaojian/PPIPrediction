import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from train_util import *
sys.path.append("./")
from ppi_network.PPIDatasetFastH5 import PPIDatasetFastH5, fast_no_go_collate
from ppi_network.PSFastPPIModel import PSFastPPIModel
from ppi_network.static_args import *

def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for residue_with_pssm, residue_with_pssm2, y in tepoch:
                output = model(toTensor(residue_with_pssm), toTensor(residue_with_pssm2))
                output = torch.round(output)
                total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
                total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def model_dir():
    return './data/PSFastPPIModels/'

def test(species):
    shuffle = False
    batch_size = 5000
    pick_num = pick_num_fast
    workers = 4
    drop_last = False
    pin_memory = True
    
    test_dataset = PPIDatasetFastH5(type='whole_' + species)
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

    # test
    total_labels,total_preds = predicting(model,test_loader)
    write_test_result_to_file(model_dir(), species, total_labels, total_preds)

if __name__ == "__main__":
    for species in test_other_species:
        test(species)