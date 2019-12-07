if __name__ == '__main__':
    input = '2	Transformer	0.1	1.00E-03	0.5	2048	2	2'
    arr = input.split()
    id = arr[0]
    lr = arr[2]
    decay = arr[3]
    dropout = arr[4]
    cuda = int(id) % 4
    cmd = 'CUDA_VISIBLE_DEVICES={cuda} python train_tf.py --tensorboard --cuda --lr={lr} --decay={decay}'.format(
        cuda=1, lr=lr, decay=decay)
    print(cmd)
