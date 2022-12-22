import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import datetime
from train_util import *
sys.path.append("./")
from ppi_network.PPIDatasetFastNoGo import PPIDatasetFastNoGo, fast_no_go_collate
from ppi_network.PSSFastPPIModel import PSSFastPPIModel
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
    return './data/PSSFastPPIModels/'

def train():
    shuffle = False
    batch_size = 2048
    pick_num = pick_num_fast
    workers = 2
    drop_last = True
    pin_memory = True
    
    train_dataset = PPIDatasetFastNoGo(type='train', pick_num=pick_num)
    train_loader = DataLoader(dataset=train_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)
    
    test_dataset = PPIDatasetFastNoGo(type='test', pick_num=pick_num)
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=drop_last,
               num_workers=workers,
               pin_memory=pin_memory,
               collate_fn=fast_no_go_collate)
    
    model = PSSFastPPIModel(batch_size=batch_size, device=device, pick_num=pick_num).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    start_epoch = -1
    newest_model_path = find_newest_model(model_dir())
    if newest_model_path != None:
        checkpoint = torch.load(model_dir() + newest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = []
    train_accs = []
    for epoch in range(100):
        if epoch <= start_epoch:
            continue
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
        
        save_model(model_dir(), epoch, model, optimizer)
        
        # test
        total_labels,total_preds = predicting(model,test_loader)
        write_acc_to_file(model_dir(), epoch, total_labels, total_preds, acc, avg_loss)

if __name__ == "__main__":
    train()