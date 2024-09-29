# 参考：https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2-%E5%9B%BE%E8%A7%A3transformer

import torch.nn as nn
import torch

class MultiheadAttention(nn.Module):
    # hid_dim: 每一个word embeding的维度
    def __init__(self,hid_dim,n_heads,dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制hid_dim必须被n_heads整除
        assert hid_dim % n_heads == 0

        # 定义W_q,W_k,W_v
        # 注意多头的计算和单头的区别在于，多头的计算是将Q,K,V分别映射到多个子空间，然后分别计算，所以计算QKV是和传统的attention是一样的，即通过一个线性变换得到Q,K,V
        self.w_q = nn.Linear(hid_dim,hid_dim)
        self.w_k = nn.Linear(hid_dim,hid_dim)
        self.w_v = nn.Linear(hid_dim,hid_dim)

        # 定义一个全连接层
        self.fc = nn.Linear(hid_dim,hid_dim)

        # 定义dropout
        self.do = nn.Dropout(dropout)

        # 定义缩放因子,用于缩放QK的点积，注意这里是多头，所以多头的维度是hid_dim//n_heads
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))


    def forward(self,query,key,value,mask=None):
        # Q,K,V的维度是[batch_size,seq_len,hid_dim]
        # 注意在seq_len上，Q，K，V可能是不一样的，也可能是一样的，比如考虑decoder块的中间的多头注意力的情况，他的K，V来自encoder的输出，而Q来自上一层的输出，所以他们的seq_len可能是不一样的

        bsz= query.shape[0]
        # 计算Q,k,V矩阵
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 将Q,K,V拆分为多头，这里拆分之后，QK的转置仍然是一个seq_len*seq_len的矩阵（这和矩阵乘法有关）
        # 多头注意力，实现上直接利用了矩阵乘法的并行性
        # 拆分之后的维度是[batch_size,n_heads,seq_len,hid_dim//n_heads]
        # view操作不会改变元素之间的顺序，只会改变形状，permute操作会改变元素的顺序，可以理解为是一个transpose操作
        Q = Q.view(bsz,-1,self.n_heads,self.hid_dim//self.n_heads).permute(0,2,1,3)
        K = K.view(bsz,-1,self.n_heads,self.hid_dim//self.n_heads).permute(0,2,1,3)
        V = V.view(bsz,-1,self.n_heads,self.hid_dim//self.n_heads).permute(0,2,1,3)

        # 计算QK的点积，注意这里是多头，所以需要在最后一个维度上进行点积
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        attention = torch.matmul(Q,K.permute(0,1,3,2))/self.scale

        # 如果mask不为空，则需要将mask中为0的位置的attention设置为一个很小的数，这样在softmax之后就会接近0
        if mask is not None:
            attention = attention.masked_fill(mask==0,-1e10)

        # 进行softmax操作，在最后一个维度上进行softmax
        attention = torch.softmax(attention,dim=-1)

        # 加上dropout
        attention = self.do(attention)

        # 进行V的加权求和，得到最终的多头注意力的输出
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]（只对最后两个维度进行矩阵乘法，前面的维度保持不变）
        x = torch.matmul(attention,V)

        # 将多头的结果拼接起来
        x = x.view(bsz,-1,self.hid_dim)
        x= self.fc(x)
        return x


# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
query = torch.rand(64, 12, 300)
# batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
key = torch.rand(64, 10, 300)
# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
value = torch.rand(64, 10, 300)
attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
output = attention(query, key, value)
## output: torch.Size([64, 12, 300])
print(output.shape)