import matplotlib.pyplot as plt


def draw_plot(train_data, test_data, train_label, test_label, title, xlabel, ylabel, epoch):
    x = range(0, epoch)
    plt.plot(x, train_data, color="blue", label=train_label, linewidth=2)
    plt.plot(x, test_data, color="orange", label=test_label, linewidth=2)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel=xlabel, fontsize=15)
    plt.ylabel(ylabel=ylabel, fontsize=15)


def draw_plots(name, train_loss_list, val_loss_list):
    epoch = len(train_loss_list)
    plt.figure(figsize=(20, 5))

    plt.subplot(2, 1, 1)
    draw_plot(train_loss_list, val_loss_list, "train_loss Line", "val_loss Line", "Loss_curve", "Epochs", "Loss", epoch)
    plt.legend()

    # plt.subplot(2, 1, 2) draw_plot(train_acc_list, val_acc_list, "train_acc Line", "val_acc Line", "Acc_curve",
    # "Epochs", "Accuracy", epoch) plt.legend()
    plt.savefig(name)