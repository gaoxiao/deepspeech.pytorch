import random

child_file = '/home/gaoxiao/code/ai_utils/experiment/child_det/d_child_2019-12-01_TO_2019-12-05.txt'
parent_file = '/home/gaoxiao/code/ai_utils/experiment/child_det/d_parent_2019-12-01_TO_2019-12-05.txt'

if __name__ == '__main__':
    with open(child_file, 'r') as f:
        child_data = f.readlines()
        print(len(child_data))
    with open(parent_file, 'r') as f:
        parent_data = f.readlines()
        print(len(parent_data))

    data = []
    for l in child_data:
        l = l.replace('/home/gaoxiao/code/ai_utils/experiment/child_det/audios/', 'data/audios/')
        data.append('{},{}\n'.format(l.strip(), 1))
    for l in parent_data:
        l = l.replace('/home/gaoxiao/code/ai_utils/experiment/child_det/audios/', 'data/audios/')
        data.append('{},{}\n'.format(l.strip(), 2))

    random.shuffle(data)
    train_ratio = 0.8
    train_splitter = int(len(data) * train_ratio)

    with open('train_manifest.csv', 'w') as f:
        f.writelines(data[:train_splitter])

    with open('val_manifest.csv', 'w') as f:
        f.writelines(data[train_splitter:])
