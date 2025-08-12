from math import sqrt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor

from transformers import BertConfig, BertModel, BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch.nn as nn
class LLMFlareNetModel(nn.Module):
    def __init__(self,args):
        super(LLMFlareNetModel, self).__init__()


        self.bert_config = BertConfig.from_pretrained(r'G:\本科\项目_比赛_论文资料\论文_LLM_VIT\bert')
        self.bert_config.num_hidden_layers = args.bert_num_hidden_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        self.tokenizer = BertTokenizer.from_pretrained(
                r'G:\本科\项目_比赛_论文资料\论文_LLM_VIT\bert',
                trust_remote_code=True,
                local_files_only=True
        )

        self.llm_model = BertModel.from_pretrained(
            r'G:\本科\项目_比赛_论文资料\论文_LLM_VIT\bert',
            trust_remote_code=True,
            local_files_only=True,
            config=self.bert_config,
        )
        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.description_data = args.description_data
        self.description_task = args.description_task
        self.reprogramming_layer = ReprogrammingLayer(args.d_model, args.n_heads, args.d_llm)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = args.num_tokens
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.patch_embedding = PatchEmbedding(args.d_model, args.patch_len, args.stride, args.dropout)

        print("初始化结束")
        self.classifier = nn.Linear(148 * 7, 2)

    def forward(self, inputs):
        # print(inputs.shape) torch.Size([16, 40, 10])

        '''
        对应论文左边提示词编码
        '''
        prompt = []
        for b in range(inputs.shape[0]):
            # min_values_str = str(min_values[b].tolist()[0])
            # max_values_str = str(max_values[b].tolist()[0])
            # median_values_str = str(medians[b].tolist()[0])
            # lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description_data}"
                f"Task description: {str(self.description_task)} ; "
                # "Input statistics: "
                # f"min value {min_values_str}, "
                # f"max value {max_values_str}, "
                # f"median value {median_values_str}, "
                # f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                # f"top 5 lags are : {lags_values_str}"
                f"<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,max_length=2048).input_ids
        # print(prompt.shape)torch.Size([16, 141])
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(inputs.device))  # (batch, prompt_token, dim)
        # print(prompt_embeddings.shape)([16, 141, 768])
        '''
        对应去所有预训练词对应含义去找跟时间有关的
        '''
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # print(self.word_embeddings.shape)torch.Size([30522, 768])
        # print(source_embeddings.shape)torch.Size([1000, 768])

        '''
        # x_enc = inputs.permute(0, 2, 1).contiguous()
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # # torch.Size([160, 8, 16])
        # # 里面把16 10 8 16前面合并了
        # 
        # 
        # enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # # print(enc_out.shape)torch.Size([160, 8, 768])
        # 
        # llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # print(llama_enc_out.shape)
        # exit()
        '''
        '''
            patch
        '''
        input_patchs,patch_num=self.patch_embedding(inputs)
        # print(x_enc.shape)torch.Size([16, 7, d_model])
        '''
            att
        '''
        input2nlp=self.reprogramming_layer(input_patchs,source_embeddings,source_embeddings)

        # print(input2nlp.shape)torch.Size([16, 7, 768])
        '''
            输入大模型
        '''
        llama_enc_out = torch.cat([prompt_embeddings, input2nlp], dim=1)
        nlp= self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state  # 7+7
        #去掉提示词，拿出数据部分长度
        nlp2data = nlp[:, :, -patch_num:]
        # print(nlp2data.shape)torch.Size([16, 148, 7])
        # 分类头
        enc_out_flat = nlp2data.reshape(nlp2data.size(0), -1)  # [16, 1036]
        attention_mul = self.classifier(enc_out_flat)
        out_put = F.log_softmax(attention_mul, dim=1)
        # out_put=1

        return out_put


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


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm,d_keys=None,attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, patch_number, D = target_embedding.shape  # (B, patch_number, d_model)
        v, _ = source_embedding.shape  # (v, d_llm), v is vocab size
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, patch_number, H, -1)  # (B, patch_number, H, d_keys)
        source_embedding = self.key_projection(source_embedding).view(v, H, -1)  # (v, H, d_keys)
        value_embedding = self.value_projection(value_embedding).view(v, H, -1)  # (v, H, d_keys)

        scale = 1. / sqrt(target_embedding.shape[-1])
        scores = torch.einsum("bphd,vhd->bhpd", target_embedding, source_embedding)  # (B, H, patch_number, v)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # (B, H, patch_number, v)
        reprogramming_embedding = torch.einsum("bhpd,vhd->bphd", A, value_embedding)  # (B, patch_number, H, d_keys)

        out = reprogramming_embedding.reshape(B, patch_number, -1)  # (B, patch_number, H * d_keys)
        out = self.out_projection(out)  # (B, patch_number, d_llm)
        return out