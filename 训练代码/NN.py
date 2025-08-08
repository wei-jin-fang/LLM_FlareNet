from math import sqrt
import torch.nn.functional as F
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel, BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        timestep=40
        num_of_classes=2
        input_size=10
        embed_dim=args.embed_dim
        self.args=args
        self.tokenbegore = nn.Linear(10, 768)
        self.tokenafter = nn.Linear(768, 10)

        self.input_projection = nn.Linear(10, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=args.num_heads, batch_first=True)
        self.input_output_projection = nn.Linear(embed_dim, input_size)  # 将 80 维映射回 10 维

        self.batch_norm = nn.BatchNorm1d(input_size)

        self.fc1 = nn.Linear(10, 100)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(100, 10)
        self.dropout2 = nn.Dropout(0.4)

        self.final_dropout = nn.Dropout(0.8)#0.7-》0.5  0.32
        # self.final_dropout = nn.Dropout(0.4)

        self.flatten = nn.Flatten()

        self.fc64 = nn.Linear(10, 64)
        self.batch_norm64 = nn.BatchNorm1d(64)
        self.fc32 = nn.Linear(64, 32)
        self.batch_norm32 = nn.BatchNorm1d(32)
        self.outlinear = nn.Linear(32, 2)

        # LSTM层
        self.lstm = nn.LSTM(input_size=10,
                            hidden_size=args.hidden_units,
                            num_layers=args.num_layers,
                            batch_first=True,  # 假设输入的第一个维度是batch size
                            bias=True)
        self.dropout4 = nn.Dropout(0.4)
        self.liner400 = nn.Linear(args.hidden_units, 400)


        print("初始化结束")

    def forward(self, inputs):
        x = inputs

        if self.args.print == 1:
            print("输出NN层")
        #拼接出输出
        # print(x.shape)
        # exit()
        x5=x  #适配输出层
        # x5 = x5.transpose(1, 2)
        x5 = self.batch_norm(x5)
        # x5 = x5.transpose(1, 2)
        attention_mul = self.final_dropout(x5)#torch.Size([16, 10])

        # attention_mul = self.flatten(attention_mul)
        attention_mul = self.fc64(attention_mul)
        attention_mul = self.batch_norm64(attention_mul)
        attention_mul = self.final_dropout(attention_mul)

        attention_mul = self.fc32(attention_mul)
        attention_mul = self.batch_norm32(attention_mul)
        attention_mul = self.final_dropout(attention_mul)

        attention_mul = self.outlinear(attention_mul)

        out_put = F.log_softmax(attention_mul, dim=1)
        if self.args.print == 1:
            self.args.print -= 1
        return out_put






