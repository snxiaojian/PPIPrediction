import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import datetime
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
        for batch_idx,(residue_with_pssm ,residue_with_pssm2, y) in enumerate(loader):
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S.%f"))
            print("predicting batch: {}".format(batch_idx))
            output = model(toTensor(residue_with_pssm), toTensor(residue_with_pssm2))
            output = torch.round(output)
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def model_dir():
    return './data/PSFastPPIModels/'

def test(test_species):
    shuffle = False
    batch_size = 5000
    pick_num = pick_num_fast
    workers = 4
    drop_last = False
    pin_memory = True
    
    test_epoch = 28
    
    model_path = model_dir() + 'epoch' + str(test_epoch) + '.pkl'
    
    test_dataset = PPIDatasetFastH5(type='test', target_species=test_species)
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)
    
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    total_labels,total_preds = predicting(model,test_loader)
    write_test_result_to_file(model_dir(), test_epoch, total_labels, total_preds, test_species)

def write_test_result_to_file(model_dir, epoch ,total_labels, total_preds, test_species):
    test_acc = accuracy_score(total_labels, total_preds)
    test_prec = precision_score(total_labels, total_preds)
    test_recall = recall_score(total_labels, total_preds)
    test_f1 = f1_score(total_labels, total_preds)
    test_auc = roc_auc_score(total_labels, total_preds)
    con_matrix = confusion_matrix(total_labels, total_preds)
    test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
    test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
    with open(model_dir  + 'epoch_' + str(epoch) + "_test_result.txt", 'a+') as fp:
        fp.write("species:" + test_species + "number:" + str(total_labels.shape[0]))
        fp.write('\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')
 
if __name__ == "__main__":
   for species in train_species:
       test(species)