import argparse 
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_file",
        help = "path to log file"
        )
    args = parser.parse_args()

    if os.path.isdir(args.log_file):
        for root, _, files in os.walk(args.log_file):
            if len(files) > 0:
                for file in files:
                    if file.find('txt') > 0:
                        print(file)
                        f = open(root+'/'+file)
                        lines = [line.rstrip("\n") for line in f.readlines()]
                        epochs = []
                        train_acc, valid_acc = [], []
                        train_loss, valid_loss = [], []

                        for line in lines:
                            try:
                                line_list = line.split(',')
                                if line_list[0][:5] == 'Epoch':
                                    a = int(line_list[0][5:])
                                    b = float(line_list[2][:-1])
                                    c = float(line_list[4])
                                    d = float(line_list[6][:-1])
                                    e = float(line_list[8])
                                    epochs.append(a)
                                    train_acc.append(b)
                                    train_loss.append(c)
                                    valid_acc.append(d)
                                    valid_loss.append(e)
                            except:
                                pass
                                print('missing a few epoch')
                        fig = plt.figure(figsize=(14, 10))
                        ax1 = fig.add_subplot(2, 1, 1)
                        ax1.plot(epochs, train_acc, 'r', label='train_acc')
                        ax1.plot(epochs, valid_acc, 'b', label='valid_acc')
                        ax1.grid()
                        ax1.title.set_text('Accuracy')
                        ax1.set_xlabel('epochs')
                        ax1.set_ylabel('accuracy %')
                        ax1.legend()
                        ax2 = fig.add_subplot(2, 1, 2)
                        ax2.plot(epochs, train_loss, 'r', label='train_loss')
                        ax2.plot(epochs, valid_loss, 'b', label='valid_loss')
                        ax2.grid()
                        ax2.title.set_text('Loss')
                        ax2.set_xlabel('epochs')
                        ax2.set_ylabel('loss')
                        ax2.legend()
                        # plt.show()
                        plt.savefig(root + '/' + file[:-3] + 'png')
                        plt.close()
    else:
        f = open(args.log_file)
        lines = [line.rstrip("\n") for line in f.readlines()]
        epochs = []
        train_acc, valid_acc = [], []
        train_loss, valid_loss = [], []

        for line in lines:
            line_list = line.split(',')
            if line_list[0][:5] == 'Epoch':
                line_list = line.split(',')
                epochs.append(int(line_list[0][5:]))
                train_acc.append(float(line_list[2][:-1]))
                train_loss.append(float(line_list[4]))
                valid_acc.append(float(line_list[6][:-1]))
                valid_loss.append(float(line_list[8]))
        # import pdb;
        # pdb.set_trace()
        fig = plt.figure(figsize=(14, 10))
        # import pdb;
        # pdb.set_trace()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(epochs, train_acc, 'r', label='train_acc')
        ax1.plot(epochs, valid_acc, 'b', label='valid_acc')
        ax1.grid()
        ax1.title.set_text('Accuracy')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy %')
        ax1.legend()
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(epochs, train_loss, 'r', label='train_loss')
        ax2.plot(epochs, valid_loss, 'b', label='valid_loss')
        ax2.grid()
        ax2.title.set_text('Loss')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('loss')
        ax2.legend()
        # plt.show()
        plt.savefig(args.log_file[:-3]+'png')
    
if __name__ == "__main__":
    main(sys.argv)