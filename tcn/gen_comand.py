if __name__ == '__main__':
    input = '4	TCN	0.1	1.00E-04	0.5	2048	7	8	450	2		0.6																						'
    arr = input.split()
    id = arr[0]
    lr = arr[2]
    decay = arr[3]
    dropout = arr[4]
    hidden_size = arr[5]
    ksize = arr[6]
    levels = arr[7]
    nhid = arr[8]
    fc_layers = arr[9]
    cuda = int(id) % 3 + 1
    cmd = 'CUDA_VISIBLE_DEVICES={cuda} python tcn/train.py ' \
          '--tensorboard --cuda --lr={lr} --decay={decay} --dropout={dropout} ' \
          '--ksize={ksize} --levels={levels} --nhid={nhid} ' \
          '--hidden-size={hidden_size} ' \
          ' --fc-layers={fc_layers} ' \
          '--id={id}'.format(
        cuda=cuda, lr=lr, decay=decay, dropout=dropout, hidden_size=hidden_size, fc_layers=fc_layers, id=id,
        ksize=ksize, levels=levels, nhid=nhid)
    print(cmd)
