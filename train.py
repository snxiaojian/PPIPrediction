import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import dgl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("./")
from ppi_network.PPIDataset import PPIDataset, collate
from ppi_network.GPSSPPIModel import GPSSPPIModel

device = torch.device('cuda')

def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx,(G_residue,go_feature, pssm, indexes ,G_residue2, go_feature2,pssm2, indexes2, y) in enumerate(loader):
            output = model(dgl.batch(G_residue), toTensor(go_feature), toTensor(pssm), toTensor(indexes),
                               dgl.batch(G_residue2), toTensor(go_feature2), toTensor(pssm2), toTensor(indexes2))
            output = torch.round(output)
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def toTensor(list_of_tensors):
    tensor_of_tensors = torch.stack((list_of_tensors))
    return tensor_of_tensors.to(device)
    
def train():

    train_losses = []
    train_accs = []
    
    train_dataset = PPIDataset(type='train')
    train_loader = DataLoader(dataset=train_dataset,
               batch_size=400,
               shuffle=True,
               drop_last=False,
               collate_fn=collate)
    
    test_dataset = PPIDataset(type='test')
    test_loader = DataLoader(dataset=test_dataset,
               batch_size=400,
               shuffle=True,
               drop_last=False,
               collate_fn=collate)
    
    model = GPSSPPIModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(100):
        total_loss = 0
        n_batches = 0
        correct = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for G_residue,go_feature, pssm, indexes ,G_residue2, go_feature2,pssm2, indexes2, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                y_pred = model(dgl.batch(G_residue), toTensor(go_feature), toTensor(pssm), toTensor(indexes),
                               dgl.batch(G_residue2), toTensor(go_feature2), toTensor(pssm2), toTensor(indexes2))
                y = y.unsqueeze(1).to(device)
                correct += torch.eq(torch.round(y_pred),y).data.sum()
                loss = criterion(y_pred,y)
                total_loss+=loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_batches+=1
                tepoch.set_postfix(loss=loss.item())
                
        avg_loss = total_loss/n_batches
        acc = correct/(len(train_loader.dataset))
        
        train_losses.append(avg_loss)
        train_accs.append(acc)
        
        print("train avg_loss is",avg_loss)
        print("train ACC = ",acc)
        
        torch.save(model.state_dict(), "./data/model_pkl/"+'epoch'+'%d.pkl'%(epoch+1))
        # test
        total_labels,total_preds = predicting(model,test_loader)
        test_acc = accuracy_score(total_labels, total_preds)
        test_prec = precision_score(total_labels, total_preds)
        test_recall = recall_score(total_labels, total_preds)
        test_f1 = f1_score(total_labels, total_preds)
        test_auc = roc_auc_score(total_labels, total_preds)
        con_matrix = confusion_matrix(total_labels, total_preds)
        test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
        test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
        print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
        with open("./data/model_pkl/result.txt", 'a+') as fp:
            fp.write('epoch:' + str(epoch+1) + '\ttrainacc=' + str(acc) +'\ttrainloss=' + str(avg_loss.item()) +'\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')

if __name__ == "__main__":
    train()