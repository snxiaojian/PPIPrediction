#以epoch为横坐标，在同一坐标下画出acc、val_acc随epoch变化的曲线图
#定义show_Training_history()函数，输入参数：训练过程所产生的Training_history
import matplotlib.pyplot as plt
import numpy as np

class TrainingHistory:
    # parse infomation from line like:
    def __init__(self, line) -> None:
        self.species = line.split()[0].split(':')[1]
        self.number = int(line.split()[1].split(':')[1])
        self.epoch = int(line.split()[2].split(':')[1])
        self.acc = float(line.split()[3].split('=')[1])
        self.prec = float(line.split()[4].split('=')[1])
        self.recall = float(line.split()[5].split('=')[1])
        self.f1 = float(line.split()[6].split('=')[1])
        self.auc = float(line.split()[7].split('=')[1])
        self.spec = float(line.split()[8].split('=')[1])
        self.mcc = float(line.split()[9].split('=')[1])

def get_training_history(model_dir):
    history_file = model_dir + '_train_epoch_result.txt'
    with open(history_file, 'r') as f:
        lines = f.readlines()
        histories = [TrainingHistory(line) for line in lines]
        return histories

def filter_histories_by_species(histories, species):
    return [h for h in histories if h.species == species]

def show_training_history(training_histories):
    human_histories = filter_histories_by_species(training_histories, 'human')
    yeast_histories = filter_histories_by_species(training_histories, 'yeast')
    fly_histories = filter_histories_by_species(training_histories, 'fly')
    arabidopsis_histories = filter_histories_by_species(training_histories, 'arabidopsis')
    all_histories = filter_histories_by_species(training_histories, 'all')
    human_acc = [h.acc for h in human_histories]
    yeast_acc = [h.acc for h in yeast_histories]
    fly_acc = [h.acc for h in fly_histories]
    arabidopsis_acc = [h.acc for h in arabidopsis_histories]
    all_acc = [h.acc for h in all_histories]
    
    lineStyle = ":"
    plt.plot(human_acc, linestyle=lineStyle, color='b')
    plt.plot(yeast_acc, linestyle=lineStyle, color='r')
    plt.plot(fly_acc, linestyle=lineStyle, color='y')
    plt.plot(arabidopsis_acc, linestyle=lineStyle, color='g')
    plt.plot(all_acc, linestyle=lineStyle, color='k')
    # 显示图的标题
    plt.title('Training accuracy history')
    # 显示x轴标签epoch
    plt.xlabel('epoch')
    # 显示y轴标签train
    plt.ylabel('train_acc')
    # 设置图例是显示'train','validation',位置在右下角
    plt.legend(['human', 'yeast', "fly", "arabidopsis", "all"], loc='lower right')
    # 开始绘图
    plt.show()
show_training_history(get_training_history('./data/PSFastPPIModels/'))