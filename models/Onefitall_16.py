from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Conv1d(
            in_channels=10,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )
        self.project_to_768 = nn.Linear(d_model, 768)  # 映射到 768 维度
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = self._init_pos_embedding(max_len=40, d_model=d_model)
        self.to(torch.float32)

    def _init_pos_embedding(self, max_len, d_model):
        # 正弦-余弦位置编码
        position = torch.arange(max_len).unsqueeze(1)  # [40, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2]
        pe = torch.zeros(max_len, d_model)  # [40, 768]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # [1, 40, 768]
        return pe  # 注册为 buffer 后可自动移到正确设备

    def forward(self, x):
        # x: [batch, 40, 10]
        B, T, N = x.shape  # T=40, N=10
        assert T % self.patch_len == 0, f"Input length {T} must be divisible by patch_len {self.patch_len}"
        x = x.permute(0, 2, 1)  # [batch, 10, 40]
        x = self.value_embedding(x)  # [batch, d_model, 40]
        x = x.permute(0, 2, 1)  # [batch, 40, d_model]

        # 添加位置编码
        pos_embedding = self.pos_embedding.to(x.device)  # [1, 40, d_model]
        x = x + pos_embedding  # [batch, 40, d_model]

        x = self.project_to_768(x)  # [batch, 40, 768]
        x =self.dropout(x)
        return x, x.shape[1]  # 返回 [batch, 40, 768], patch_num=40


class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()
        self.flatten = nn.Flatten()
        self.direct1 = None
        self.final_dropout=nn.Dropout(args.dropout_rate)
        self.output_dim=args.output_dim

    def forward(self, x):
        # x 的形状: [batch_size, patch_num, nlp_last_dim]，例如 [16, 40, 768]
        # 动态获取 nlp_last_dim 和 patch_num
        batch_size, patch_num, nlp_last_dim = x.shape
        if self.direct1 is None:
            input_dim = nlp_last_dim * patch_num
            self.direct1 = nn.Linear(input_dim, self.output_dim).to(x.device)
        x = self.flatten(x)  # 形状: [batch_size, patch_num * nlp_last_dim]

        x = self.direct1(x)
        x = self.final_dropout(x)
        # 形状: [batch_size, output_dim]

        return x

class Onefitall_16Model(nn.Module):
    def __init__(self, args):
        super(Onefitall_16Model, self).__init__()

        self.bert_config = BertConfig.from_pretrained(r'E:\conda_code_tf\LLM\bert')
        self.bert_config.num_hidden_layers = args.bert_num_hidden_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        self.llm_model = BertModel.from_pretrained(
            r'E:\conda_code_tf\LLM\bert',
            trust_remote_code=True,
            local_files_only=True,
            config=self.bert_config,
        )
        # 只解冻LayerNorm参与训练
        for name, param in self.llm_model.named_parameters():
            if "LayerNorm" in name:
                param.requires_grad = True

            else:
                param.requires_grad = False

        self.d_model = args.d_model
        self.patch_embedding = PatchEmbedding(args.d_model, patch_len=1, stride=1, dropout=args.dropout)
        self.classification_head = ClassificationHead(args)
        # 添加 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        print("初始化结束")

    def forward(self, inputs):
        # inputs: [batch, 40, 10]
        # Patch 嵌入
        input_patchs, patch_num = self.patch_embedding(inputs)  # [batch, 40, d_model]
        # print(input_patchs.shape,patch_num)
        # torch.Size([16, 40, 768]) 40
        # ReprogrammingLayer 处理

        # 输入 BERT 模型
        nlp = self.llm_model(inputs_embeds=input_patchs).last_hidden_state  # [batch, 40, 768]
        # 分类头
        x = self.classification_head(nlp)
        # 添加 Sigmoid 激活函数
        x = self.sigmoid(x)  # 形状: [batch_size, 1]
        return x

