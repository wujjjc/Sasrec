import torch
import torch.nn as nn


class Embedding_layer(nn.Module):
    def __init__(self, embedding_num, embedding_size):
        super(Embedding_layer, self).__init__()
        self.embedding = nn.Embedding(embedding_num, embedding_size, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


class Self_Attention(nn.Module):
    def __init__(self, embedding_size):
        super(Self_Attention, self).__init__()
        self.embedding_size = embedding_size
        self.q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v = nn.Linear(embedding_size, embedding_size, bias=False)
        self.soft = nn.Softmax(dim=-1)


    def forward(self, x, mask):
        Q = self.q(x) 
        K = self.k(x)
        V = self.v(x)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embedding_size ** 0.5) #[batch, n, n]
        seq_len = scores.shape[1]
        # 仅屏蔽未来位置(上三角，不含对角线)
        mask_time = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        #padding填充
        if mask is not None:
            # key维padding屏蔽
            padding_mask = ~(mask.unsqueeze(1)) #取反
            scores = scores.masked_fill(padding_mask, float('-inf')) #填充
            # query维padding位置不参与注意力，避免全-inf行导致softmax出现NaN
            query_padding_mask = ~(mask.unsqueeze(-1))
            scores = scores.masked_fill(query_padding_mask, 0.0)
        scores = scores.masked_fill(mask_time.unsqueeze(0), float('-inf')) #去除时间因果性
        attention_weights = self.soft(scores)
        output = torch.matmul(attention_weights, V)
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        return output


class FFN(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(FFN, self).__init__()
        self.embedding_size = embedding_size
        self.linear1 = nn.Linear(embedding_size, embedding_size, bias=True)
        self.linear2 = nn.Linear(embedding_size, embedding_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #x:[batch, n, embedding_size]
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x #x : [batch, n, embedding_size]


class Sasrec_Net(nn.Module):
    def __init__(self, embedding_size, embedding_num, max_len, mode=None, dropout=0.2):
        super(Sasrec_Net, self).__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.embedding = Embedding_layer(embedding_num, embedding_size)
        self.pos_embedding = nn.Embedding(max_len, embedding_size)
        self.self_attention1 = Self_Attention(embedding_size)
        self.self_attention2 = Self_Attention(embedding_size)
        self.ffn1 = FFN(embedding_size, dropout)
        self.ffn2 = FFN(embedding_size, dropout)
        self.mode = mode
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.layer_norm3 = nn.LayerNorm(embedding_size)
        self.layer_norm4 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)



    def residual_apply(self, x, net, layer_norm,mask=None):
        residual = x
        x = layer_norm(x)  # 正则化
        if mask is not None: #attention层
            return residual + (self.dropout(net(x, mask))) #残差连接
        return residual + (self.dropout(net(x))) #残差连接


    def forward(self, item_seq, mask, neg_item=None, pos_item=None): #item_train表示正负样本的连接
        if self.mode == 'item': #返回物品嵌入
            return self.embedding(item_seq)
        seq_len = item_seq.shape[1]
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)  # (1, seq_len), 位置编码
        embedding = self.dropout(self.embedding(item_seq) + self.pos_embedding(positions))
        embedding *= mask.unsqueeze(-1).to(embedding.dtype) #padding位置置0
        attention1 = self.residual_apply(embedding, self.self_attention1, self.layer_norm1, mask)
        attention1 *= mask.unsqueeze(-1).to(attention1.dtype) #padding位置置0
        ffn1 = self.residual_apply(attention1, self.ffn1, self.layer_norm2)
        ffn1 *= mask.unsqueeze(-1).to(ffn1.dtype) #padding位置置0
        attention2 = self.residual_apply(ffn1, self.self_attention2, self.layer_norm3, mask)
        attention2 *= mask.unsqueeze(-1).to(attention2.dtype) #padding位置置0
        output = self.residual_apply(attention2, self.ffn2, self.layer_norm4)
        output *= mask.unsqueeze(-1).to(output.dtype)
        output = output[:, -1:, :] #batch, 1, emb
        if self.mode == 'user':
            return output.squeeze(1) #batch,1
        return torch.matmul(output, self.embedding(torch.cat([pos_item, neg_item], dim=-1)).transpose(-1, -2)).squeeze(1) #batch, n














