if __name__ == '__main__':
    input = '21	Transformer	0.03	1.00E-04	0.7	512	2	2	16	15																							'
    arr = input.split()
    id = arr[0]
    lr = arr[2]
    decay = arr[3]
    dropout = arr[4]
    hidden_size = arr[5]
    hidden_layers = arr[6]
    fc_layers = arr[7]
    decoder_output = arr[8]
    cuda = int(id) % 3 + 1
    cmd = 'CUDA_VISIBLE_DEVICES={cuda} python transformer/train.py ' \
          '--tensorboard --cuda --lr={lr} --decay={decay} --dropout={dropout} ' \
          '--hidden-layers={hidden_layers} --hidden-size={hidden_size} ' \
          '--tf-decoder-output={decoder_output} --fc-layers={fc_layers} ' \
          '--id={id}'.format(
        cuda=cuda, lr=lr, decay=decay, dropout=dropout,
        hidden_layers=hidden_layers, hidden_size=hidden_size,
        decoder_output=decoder_output, fc_layers=fc_layers, id=id)
    print(cmd)
