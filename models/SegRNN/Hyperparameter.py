

H = 400  # history/input window

L = 200  # forecast/prediction window

batch_size = 32
epochs = 100
lr = 0.0005
fs = 10 

class Configs:
    def __init__(self):
        self.L = H                 # model input/history length
        self.H = L                 # model forecast/prediction length
        self.enc_in = 1           # 杈撳叆缁村害
        self.num_layer = 32
        self.dropout = 0.1        # dropout 姒傜巼
        self.rnn_type = 'rnn'    # RNN绫诲瀷
        self.dec_way = 'rmf'      # 瑙ｇ爜鏂瑰紡
        self.seg_len = 10          # 鐗囨闀垮害
        self.channel_id = False   # 鏄惁鍚敤channel id
        self.revin = True         # 鏄惁浣跨敤RevINN

use_early_stopping = False
early_stopping_patience = 5

# Unified experiment overrides: H=input history length, L=forecast length.
import os
H = int(os.getenv('EXP_H', H))
L = int(os.getenv('EXP_L', L))


