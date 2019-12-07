CUDA_VISIBLE_DEVICES=2 python train.py --tensorboard --cuda --hidden-layers=1 --model-path=models/deepspeech_2.pth
CUDA_VISIBLE_DEVICES=2 python train.py --tensorboard --cuda --hidden-layers=1 --model-path=models/deepspeech_2_3_fc.pth

CUDA_VISIBLE_DEVICES=1 python train.py --tensorboard --cuda --hidden-layers=5 --model-path=models/deepspeech_1_5rnn_3_fc.pth


# Dropout doesn't help.
CUDA_VISIBLE_DEVICES=1 python train.py --tensorboard --cuda --hidden-layers=3 --model-path=models/deepspeech_1.pth
CUDA_VISIBLE_DEVICES=1 python train.py --tensorboard --cuda --hidden-layers=3 --model-path=models/deepspeech_1.pth



CUDA_VISIBLE_DEVICES=3 python test.py --cuda --model-path=models/deepspeech_2.pth
CUDA_VISIBLE_DEVICES=3 python test.py --cuda --model-path=models/deepspeech_2_3_fc.pth


RNN 前后向 hidden 不加起来？
RNN 添加attention？
RNN 利用output 信息？
