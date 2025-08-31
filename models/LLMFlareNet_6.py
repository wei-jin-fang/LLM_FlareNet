from math import sqrt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor

from transformers import BertConfig, BertModel, BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch.nn as nn


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


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Conv1d(in_channels=10, out_channels=d_model,
                                        kernel_size=patch_len, stride=stride)
        self.dropout = nn.Dropout(dropout)
        self.to(torch.float32)  # 确保权重和偏置是 float32

    def forward(self, x):
        # x: [B, T, N], 例如 [16, 40, 10]
        B, T, N = x.shape
        assert T % self.patch_len == 0, f"Input length {T} must be divisible by patch_len {self.patch_len}"
        x = x.permute(0, 2, 1)  # [B, N, T], 例如 [16, 10, 40]
        x = self.value_embedding(x)  # [B, d_model, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, d_model]
        return self.dropout(x), x.shape[1]  # 返回嵌入和 patch 数量




class LLMFlareNet_6Model(nn.Module):
    def __init__(self,args):
        super(LLMFlareNet_6Model, self).__init__()


        # self.bert_config = BertConfig.from_pretrained(r'E:\conda_code_tf\LLM\bert')
        # self.bert_config.num_hidden_layers = args.bert_num_hidden_layers
        # self.bert_config.output_attentions = True
        # self.bert_config.output_hidden_states = True
        #
        # self.tokenizer = BertTokenizer.from_pretrained(
        #         r'E:\conda_code_tf\LLM\bert',
        #         trust_remote_code=True,
        #         local_files_only=True
        # )
        #
        # self.llm_model = BertModel.from_pretrained(
        #     r'E:\conda_code_tf\LLM\bert',
        #     trust_remote_code=True,
        #     local_files_only=True,
        #     config=self.bert_config,
        # )
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        # self.description_data = args.description_data
        # self.description_task = args.description_task
        # self.reprogramming_layer = ReprogrammingLayer(args.d_model, args.n_heads, args.d_llm)
        # self.word_embeddings = self.llm_model.get_input_embeddings().weight
        # self.vocab_size = self.word_embeddings.shape[0]
        # self.num_tokens = args.num_tokens
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.patch_embedding = PatchEmbedding(args.d_model, args.patch_len, args.stride, args.dropout)
        print("初始化结束")
        # 分类头
        self.classification_head = ClassificationHead(args)
        # 添加 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):

        '''
            patch
        '''
        input_patchs,patch_num=self.patch_embedding(inputs)
        # print(input_patchs.shape)##torch.Size([16, 64, d_model])
        out_put = self.classification_head(input_patchs)
        # 添加 Sigmoid 激活函数
        x = self.sigmoid(out_put)  # 形状: [batch_size, 1]
        return x

