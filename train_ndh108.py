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
from itertools import chain
torch.multiprocessing.set_sharing_strategy('file_system')

from enum import Enum

class TrainType(Enum):
    FULL = 1
    ONLY_FC = 2 
    ONLY_OUT = 3
    FC_AND_OUT = 4
    MODIFYFC_AND_OUT = 5
    

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

def model_dir_for(target_species, train_type):
    return './data/PSFastPPIModels'

def train(target_species, train_type = TrainType.FULL):
    shuffle = False
    batch_size = 1000
    pick_num = pick_num_fast
    workers = 2
    drop_last = False
    pin_memory = False
    
    train_dataset = PPIDatasetFastH5(type='train', target_species = target_species)
    train_loader = DataLoader(dataset=train_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)
    
    test_dataset = PPIDatasetFastH5(type='test', target_species = target_species)
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)
    # print train_dataset number of samples
    print("train_dataset number of samples: ", len(train_dataset))
    print("test_dataset number of samples: ", len(test_dataset))
    model_dir = model_dir_for(target_species, train_type=train_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = model_for_train(pick_num=pick_num, train_type=train_type)
    optimizer = optimizer_for_train(model=model, train_type=train_type)
    criterion = torch.nn.BCELoss()
 
    train_losses = []
    train_accs = []
    for epoch in range(500):
        total_loss = 0
        n_batches = 0
        correct = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for residue_with_pssm, residue_with_pssm2, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                model.train()
                y_pred = model(toTensor(residue_with_pssm), toTensor(residue_with_pssm2))
                y = y.unsqueeze(1).to(device)
                correct_of_this_batch = torch.eq(torch.round(y_pred),y).data.sum()
                correct += correct_of_this_batch
                loss = criterion(y_pred,y)
                total_loss+=loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_batches+=1
                tepoch.set_postfix(loss=loss.item(), correct=correct_of_this_batch.item()/batch_size)
                
        avg_loss = total_loss/n_batches
        acc = correct/(len(train_loader.dataset))
        
        train_losses.append(avg_loss)
        train_accs.append(acc)
        
        print("train avg_loss is",avg_loss)
        print("train ACC = ",acc)
        save_model(model_dir, epoch, model, optimizer)
        
        # test
        total_labels,total_preds = predicting(model,test_loader)
        write_acc_to_file(model_dir, epoch, total_labels, total_preds, acc.data, avg_loss)
def model_for_train(pick_num, train_type):
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)
    model_path = './data/PSFastPPIModels/' + 'epoch' + "62" + '.pkl'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if train_type == TrainType.ONLY_FC:
        for param in model.parameters():
            param.requires_grad = False 
        for param in model.fc.parameters():
            param.requires_grad = True

    elif train_type == TrainType.ONLY_OUT:
        for param in model.parameters():
            param.requires_grad = False 
        for param in model.out.parameters():
            param.requires_grad = True
    
    elif train_type == TrainType.FC_AND_OUT:
        for param in model.parameters():
            param.requires_grad = False 
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.out.parameters():
            param.requires_grad = True
    elif train_type == TrainType.MODIFYFC_AND_OUT:
        for param in model.parameters():
            param.requires_grad = False 
        model.fc = torch.nn.Linear(model.graph_pssm_output_dim*2, 256, device=device)
        model.out = torch.nn.Linear(256, 1, device=device)
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model

def optimizer_for_train(model, train_type):
    if train_type == TrainType.ONLY_FC:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    elif train_type == TrainType.ONLY_OUT:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=0.001)
    elif train_type == TrainType.FC_AND_OUT:
        optimizer = torch.optim.Adam(chain(model.fc.parameters(), model.out.parameters()), lr=0.001)
    elif train_type == TrainType.MODIFYFC_AND_OUT:
        optimizer = torch.optim.Adam(chain(model.fc.parameters(), model.out.parameters()), lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer
    
def load_pretrained_model(pick_num):
    model_path = './data/PSFastPPIModels/' + 'epoch' + "12" + '.pkl'
    model = PSFastPPIModel(device=device, pick_num=pick_num).to(device=device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Freeze the weights of the earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last linear layer for new dataset
    # model.fc = torch.nn.Linear(model.graph_pssm_output_dim*2, 32, device=device)
    # model.out = torch.nn.Linear(32, 1, device=device) 
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    for param in model.out.parameters():
        param.requires_grad = True
        
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print(f"Total number of parameters: {num_params}")
    
    num_params = 0
    for param in model.fc.parameters():
        num_params += param.numel()

    print(f"Total number of fc parameters: {num_params}")
    
    num_params = 0
    for param in model.out.parameters():
        num_params += param.numel()

    print(f"Total number of out parameters: {num_params}")
    return model

if __name__ == "__main__":
    # train("NDH108", train_type=TrainType.FULL)
    # train("NDH108", train_type=TrainType.ONLY_FC)
    # train("NDH108", train_type=TrainType.ONLY_OUT)
    # train("NDH108", train_type=TrainType.FC_AND_OUT)
    load_pretrained_model(pick_num=50)