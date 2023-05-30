#以epoch为横坐标，在同一坐标下画出acc、val_acc随epoch变化的曲线图
#定义show_Training_history()函数，输入参数：训练过程所产生的Training_history
import matplotlib.pyplot as plt
import matplotlib

class TrainingHistory:
    # parse infomation from line like:
    def __init__(self, line) -> None:
        self.species = line.split()[0].split(':')[1]
        self.number = int(line.split()[1].split(':')[1])
        self.epoch = int(line.split()[2].split(':')[1])
        self.mode = line.split()[3].split(':')[1]
        self.acc = float(line.split()[4].split('=')[1])
        self.prec = float(line.split()[5].split('=')[1])
        self.recall = float(line.split()[6].split('=')[1])
        self.f1 = float(line.split()[7].split('=')[1])
        self.auc = float(line.split()[8].split('=')[1])
        self.spec = float(line.split()[9].split('=')[1])
        self.mcc = float(line.split()[10].split('=')[1])
        self.data = {'acc': self.acc, 'prec': self.prec, 'recall': self.recall, 'f1': self.f1, 'auc': self.auc, 'spec': self.spec, 'mcc': self.mcc}


def get_training_history(model_dir):
    history_file = model_dir + 'train_epoch_result.txt'
    with open(history_file, 'r') as f:
        lines = f.readlines()
        histories = [TrainingHistory(line) for line in lines]
        return histories

def filter_histories_by_species(histories, species, mode):
    return [h for h in histories if h.species == species and h.mode == mode]

def show_training_history(ax,training_histories, key, yLabel, mode):
    human_histories = filter_histories_by_species(training_histories, 'human', mode)
    yeast_histories = filter_histories_by_species(training_histories, 'yeast', mode)
    fly_histories = filter_histories_by_species(training_histories, 'fly', mode)
    arabidopsis_histories = filter_histories_by_species(training_histories, 'arabidopsis', mode)
    all_histories = filter_histories_by_species(training_histories, 'all', mode)
    human_acc = [h.data[key] for h in human_histories]
    yeast_acc = [h.data[key] for h in yeast_histories]
    fly_acc = [h.data[key] for h in fly_histories]
    arabidopsis_acc = [h.data[key] for h in arabidopsis_histories]
    all_acc = [h.data[key] for h in all_histories]
    
    lineStyle = "-"
    marker = 'o'
    marker_size = 2
    ax.plot(human_acc, color='b',linestyle=lineStyle, marker=marker, markersize=marker_size)
    ax.plot(yeast_acc, linestyle=lineStyle, color='r', marker=marker, markersize=marker_size)
    ax.plot(fly_acc, linestyle=lineStyle, color='y',marker=marker, markersize=marker_size)
    ax.plot(arabidopsis_acc, linestyle=lineStyle, color='g', marker=marker, markersize=marker_size)
    ax.plot(all_acc, linestyle=lineStyle, color='k', marker=marker, markersize=marker_size)
    # 显示图的标题
    title = mode.capitalize() + "ing " + yLabel + " History"
    ax.set_title(title)
    # 显示x轴标签epoch
    ax.set_xlabel('epoch')
    # 显示y轴标签train
    ax.set_ylabel(yLabel)
    # 设置图例是显示'train','validation',位置在右下角
    ax.legend(['human', 'yeast', "fly", "arabidopsis", "all"], loc='lower right')

def plot(mode):
    fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), dpi=100)
    ax1=axes[0,0]
    ax2=axes[0,1]
    ax3=axes[1,0]
    ax4=axes[1,1]
    show_training_history(ax1,get_training_history('./data/PSFastPPIModels/'), 'acc', 'Accuracy', mode=mode)
    show_training_history(ax2,get_training_history('./data/PSFastPPIModels/'), 'prec', 'Precision', mode=mode)
    show_training_history(ax3,get_training_history('./data/PSFastPPIModels/'), 'recall', 'Recall', mode=mode)
    show_training_history(ax4,get_training_history('./data/PSFastPPIModels/'), 'f1', 'F1-Score', mode=mode)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(mode+'.png')
plot("train")
plot("test")