import random

raw_file = '/home/xiao/code/ai_utils/experiment/child_det/d_2019-12-01_TO_2019-12-05.txt'

if __name__ == '__main__':
    with open(raw_file, 'r') as f:
        data = f.readlines()
        print(len(data))

    random.shuffle(data)
    train_ratio = 0.8
    train_splitter = int(len(data) * train_ratio)

    with open('train_manifest.csv', 'w') as f:
        f.writelines(data[:train_splitter])

    with open('val_manifest.csv', 'w') as f:
        f.writelines(data[train_splitter:])
