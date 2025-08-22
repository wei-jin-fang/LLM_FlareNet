from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class OnefitallModel(nn.Module):
    def __init__(self, args):
        super(OnefitallModel, self).__init__()

        self.bert_config = BertConfig.from_pretrained(r'./pre_train_model/bert')
        self.bert_config.num_hidden_layers = args.bert_num_hidden_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        self.llm_model = BertModel.from_pretrained(
            r'./pre_train_model/bert',
            trust_remote_code=True,
            local_files_only=True,
            config=self.bert_config,
        )
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.d_model = args.d_model
        self.patch_embedding = PatchEmbedding(args.d_model, patch_len=1, stride=1, dropout=args.dropout)
        self.classifier = nn.Linear(40 * 768, 2)  # 40 patches, each 768 dim

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
        enc_out_flat = nlp.reshape(nlp.size(0), -1)  # [batch, 40 * 768]
        attention_mul = self.classifier(enc_out_flat)
        out_put = F.log_softmax(attention_mul, dim=1)

        return out_put


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
        return self.dropout(x), x.shape[1]  # 返回 [batch, 40, 768], patch_num=40


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding):
        B, patch_number, D = target_embedding.shape  # [batch, 40, d_model]
        H = self.n_heads

        # 自注意力机制
        query = self.query_projection(target_embedding).view(B, patch_number, H, -1)  # [batch, 40, H, d_keys]
        key = self.key_projection(target_embedding).view(B, patch_number, H, -1)  # [batch, 40, H, d_keys]
        value = self.value_projection(target_embedding).view(B, patch_number, H, -1)  # [batch, 40, H, d_keys]

        scale = 1. / sqrt(query.shape[-1])
        scores = torch.einsum("bphd,bqhd->bhpq", query, key)  # [batch, H, 40, 40]
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [batch, H, 40, 40]
        reprogramming_embedding = torch.einsum("bhpq,bqhd->bphd", A, value)  # [batch, 40, H, d_keys]

        out = reprogramming_embedding.reshape(B, patch_number, -1)  # [batch, 40, H * d_keys]
        out = self.out_projection(out)  # [batch, 40, d_llm=768]
        return out