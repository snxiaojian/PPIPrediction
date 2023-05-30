import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_newest_model(model_dir):
    files = []
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            files.append(file)
    files.sort(key=lambda x: get_number(x), reverse=True)
    if len(files) == 0:
        return None
    return files[0]
# get number in string
def get_number(s):
    return int(s.removeprefix("epoch").removesuffix(".pkl"))

def save_model(model_dir, epoch, model, optimizer):
    save_path = model_dir +'epoch'+'%d.pkl'%(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

def toTensor(list_of_tensors):
    tensor_of_tensors = torch.stack((list_of_tensors))
    return tensor_of_tensors.to(device)

def write_acc_to_file(model_dir, epoch ,total_labels, total_preds, acc, avg_loss):
    test_acc = accuracy_score(total_labels, total_preds)
    test_prec = precision_score(total_labels, total_preds)
    test_recall = recall_score(total_labels, total_preds)
    test_f1 = f1_score(total_labels, total_preds)
    test_auc = roc_auc_score(total_labels, total_preds)
    con_matrix = confusion_matrix(total_labels, total_preds)
    test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
    test_mcc = matthews_corrcoef(con_matrix)
    print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
    with open(model_dir + "result.txt", 'a+') as fp:
        fp.write('epoch:' + str(epoch+1) + '\ttrainacc=' + str(acc) +'\ttrainloss=' + str(avg_loss.item()) +'\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')
        
def write_test_result_to_file(model_dir, species ,total_labels, total_preds):
    test_acc = accuracy_score(total_labels, total_preds)
    test_prec = precision_score(total_labels, total_preds)
    test_recall = recall_score(total_labels, total_preds)
    test_f1 = f1_score(total_labels, total_preds)
    test_auc = roc_auc_score(total_labels, total_preds)
    con_matrix = confusion_matrix(total_labels, total_preds)
    test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
    test_mcc = matthews_corrcoef(con_matrix)
    print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
    with open(model_dir + "test_result.txt", 'a+') as fp:
        fp.write('species:' + species + '\tnum=' + str(len(total_labels)) +  '\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) +  '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec='+str(test_spec)+ '\tmcc='+str(test_mcc)+'\n')

def matthews_corrcoef(conf_matrix):
    true_pos = conf_matrix[1,1]
    false_pos = conf_matrix[1,0]
    false_neg = conf_matrix[0,1]
    n_points = conf_matrix.sum()*1.0
    pos_rate = (true_pos + false_neg) / n_points
    activity = (true_pos + false_pos) / n_points
    mcc_numerator = true_pos / n_points - pos_rate * activity
    mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
    return mcc_numerator / numpy.sqrt(mcc_denominator)