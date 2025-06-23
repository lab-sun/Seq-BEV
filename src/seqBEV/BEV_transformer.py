import copy
#from turtle import forward
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


import sys
sys.path.append('/workspace/')
from src.seqBEV.position_encoding import build_position_encoding
from configs.opt import get_args

verbose = False

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, 
                 num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """parameters:
            d_model: 传入的为hidden_dim
            nhead: 多头注意力的个数
            num_encoder_layers: transformer中encoder的block数
            num_decoder_layers: transformer中decoder的block数
            dim_feedforward: Intermediate size of the feedforward layers in the transformer blocks
            dropout: dropout rate
            activation: 激活函数
            normalize_before: 是否在forward之前对输入normalize 
            return_intermediate_dec: 在decoder中时候记录每一层的output，若为ture，则返回num_decoder_layers个output
           -----------
           functions:
            _reset_parameters: 初始化模型权重
        """
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 1.let see the shape of input src:{}, the shape of query_embed:{}, and the shape of pos_embed:{}, the shape of mask:{}".format(src.shape, query_embed.shape, pos_embed.shape, mask.shape))
        bs, c, h, w = src.shape  # 在传入之前已经将src变成了[bs, hidden_dim, h, w]维， 即c=hidden_dim
        src = src.flatten(2).permute(2, 0, 1)   # 将[bs, c, h, w]维的src变成[h*w, bs, c]
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 2.after flatten and permute, let see the shape of input src:{}".format(src.shape))
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # 将[bs, c, h, w]维的pos_embed变成[h*w, bs, c]
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 3.after flatten and permute, let see the shape of pos_embed:{}".format(pos_embed.shape))
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 将[num_queries, hidden_dim]维的query_embed变成[num_queries, bs, hidden_dim]。num_queries为词向量的个数，hidden_dim为词向量的长度
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 4.after unsqueeze and repeat, let see the shape of query_embed:{}".format(query_embed.shape))
        mask = mask.flatten(1)  # 将[bs, h, w]维的mask变成[bs, h*w]
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 5.after mask.flatten, let see the shape of mask:{}".format(mask.shape))

        tgt = torch.zeros_like(query_embed)  # transformer decoder 输出和query_embed维度相同, 为[num_queries, bs, hidden_dim]
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 6.let see the shape of tgt:{}".format(tgt.shape))
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # memory.shape: ([h*w, bs, hidden_dim])  是encoder和decoder之间的注意力传递？
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 7.after encoder let see the shape of memory:{}".format(memory.shape))
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # return_intermediate_dec=True 则 hs shape: [num_decoder_layers, num_queries, bs, hidden_dim]， else hs shape: [1, num_queries, bs, hidden_dim]
        if verbose: print("***in BEV_transformer, TransformerEncoderLayer, forward, 8.after decoder let see the shape of hs:{}".format(hs.shape))
        # 返回hs.transpose(1,2)为[num_decoder_layers, bs, num_queries, hidden_dim], memory shape [bs, hidden_dim, h, w]
        return hs.transpose(1,2), memory.permute(1,2,0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # 复制num_layers层encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor]=None,
                src_key_padding_mask: Optional[Tensor]=None,
                pos: Optional[Tensor]=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor]=None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos:Optional[Tensor] = None):
        output = tgt
        if verbose: print(">>in BEV_transformer, TransformerDecoder, output.shape: ", output.shape)

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            #print(">>>>>in BEV_transformer, TransformerDecoder, output.shape after decoder: ", torch.stack(intermediate).shape)
            return torch.stack(intermediate)

        #print(">>>>>in BEV_transformer, TransformerDecoder, output.shape after decoder: ", output.shape)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)   #注意 d_model要整除nhead，
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor]=None,
                     src_key_padding_mask: Optional[Tensor]=None,
                     pos: Optional[Tensor]=None):
        #print("in BEV_transformer, TransformerEncoderLayer, forward post, let see the shape of input src:{}, and the shape of pos:{}".format(src.shape, pos.shape))
        q = k = self.with_pos_embed(src, pos)  # 将输入和位置编码相加，得到相同的q，k。其中src是输入经过linear得到的（是否linear的过程就是输入学习kq的过程？但为什么q=k？q，k与v的区别只在于又没有加上pos？）。
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,    # 向nn.MultiheadAttention中传入q,k,v。其中q=k=src+pos，v=src
                              key_padding_mask=src_key_padding_mask)[0]  # ？？经过self.self_attn后得到的src2是代表什么？是否是已经normalize后的attention score,还是attention score与value相乘后的结果？
                                                                         # 是attention score与value相乘后的结果，nn.MultiheadAttention返回两个值：attn_output（其形状为(L,N,E)，是在value上乘weight），attn_output_weights（形状是(N,L,L)，是每个单词与其他单词产生的weight）
                                                                         # 输出的attn_output经过linear后，作为下一个encoder_layer的输入（即src）
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor]=None,
                src_key_padding_mask: Optional[Tensor]=None,
                pos: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feeddorward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor]=None,
                     memory_mask: Optional[Tensor]=None,
                     tgt_key_padding_mask: Optional[Tensor]=None,
                     memory_key_padding_mask: Optional[Tensor]=None,
                     pos: Optional[Tensor]=None, 
                     query_pos: Optional[Tensor]=None):
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 1.let see the shape of input tgt:{}, and the shape of momery:{}".format(tgt.shape, memory.shape))
        q = k = self.with_pos_embed(tgt, query_pos)  # q = k = tgt+query_pos, 其中tgt初始化为全零的tensor，query_pos为通过nn.Embedding后得到的query_embedding(lookup table)的权重。query_pos值在decoder中用了
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 2.let see the shape of q:{}, and the shape of k:{}".format(q.shape, k.shape))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # 计算self-attention
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 3.after self attention let see the shape of tgt2:{}".format(tgt2.shape))
        tgt = tgt + self.dropout1(tgt2)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 4.after dropout tgt2 let see the shape of tgt:{}".format(tgt.shape))
        tgt = self.norm1(tgt)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 5.after norm let see the shape of tgt:{}".format(tgt.shape))
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  # 计算cross-attention， 其中q为自注意力中的tgt+query_pos，k为从encoder中得到的memory+pos，v为从encoder中得到的memory
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 6.after multihead_attn let see the shape of tgt2:{}".format(tgt2.shape))
        tgt = tgt + self.dropout2(tgt2)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 7.after dropout tgt2 let see the shape of tgt:{}".format(tgt.shape))
        tgt = self.norm2(tgt)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 8.after norm2 let see the shape of tgt:{}".format(tgt.shape))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 9.after linear2 let see the shape of tgt2:{}".format(tgt2.shape))
        tgt = tgt + self.dropout3(tgt2)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 10.after dropout3 let see the shape of tgt:{}".format(tgt.shape))
        tgt = self.norm3(tgt)
        #print("in BEV_transformer, TransformerDecoderLayer, forward post, 11.after norm3 let see the shape of tgt:{}".format(tgt.shape))
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor]=None,
                    memory_mask: Optional[Tensor]=None,
                    tgt_key_padding_amsk: Optional[Tensor]=None,
                    memory_key_padding_mask: Optional[Tensor]=None,
                    pos: Optional[Tensor]=None,
                    query_pos: Optional[Tensor]=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_amsk)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_maskk=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor]=None,
                memory_mask: Optional[Tensor]=None,
                tgt_key_padding_mask: Optional[Tensor]=None,
                memory_key_padding_mask: Optional[Tensor]=None,
                pos: Optional[Tensor]=None,
                query_pos: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#def build_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm):
def build_transformer(args, num_queries, backbone_num_channels):
    query_embed = nn.Embedding(num_queries, args.hidden_dim)  # object queires
    input_proj = nn.Conv2d(backbone_num_channels, args.hidden_dim, kernel_size=1)
    position_embedding = build_position_encoding(hidden_dim=256, position_embedding=args.position_embedding)
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    return query_embed, input_proj, position_embedding, transformer

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should br relu/gelu, not {activation}")

############################################transformer for segmentation####################################################################
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """
    def __init__(self, dim, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4]
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        #self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        #self.gn3 = nn.GroupNorm(8, inter_dims[3])

        self.out_lay = torch.nn.Conv2d(inter_dims[1], 1, 3, padding=1)

        self.dim = dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, object_mask: Tensor):
        x = torch.cat([_expand(x, object_mask.shape[1]), object_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        #print("in MaskHeadSmallConv, x shape after lay1: ", x.shape)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        #print("in MaskHeadSmallConv, x shape after lay2: ", x.shape)
        x = self.gn2(x)
        x = F.relu(x)

        x = self.out_lay(x)
        #print("in MaskHeadSmallConv, x shape after out_lay: ", x.shape)
        return x
#####################################Build DETR###########################################################

class build_obj_transformer(nn.Module):
    """This is modified DETR module"""
    def __init__(self, args, encoded_feature_chnls, num_queries, embed_method):
        """Initializes the model,
        Parameters:
            encoded_feature: the feature got from TS_encoder 
            transformer: torch module of the transformer architecture
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            embed_method: the position embedding method: sine or learned
        """
        super().__init__()
        self.num_queries = num_queries
        encoded_feature_num_channels = encoded_feature_chnls
        self.embed_method = embed_method
        self.class_embed = None  # FIXME
        self.bbox_embed = None  # FIXME
        # query_embed 通过 nn.Embedding 初始化为一个随机的查找表，字典长度为num_queries， 词向量维度为hidden_dim；查找时，输入indices，返回对应词向量
        # input_proj 将输入的encoded_feature的通道数调整为hidden_dim维
        self.query_embed, self.input_proj, self.position_embedding, self.trans_model = build_transformer(args, num_queries=self.num_queries, backbone_num_channels=encoded_feature_num_channels)

        #self.encoded_feature = encoded_feature  # 由TS_encoder以及TS_fusion模块处理后，得到的特征张量
        self.aux_loss = None # FIXME
        hidden_dim = self.trans_model.d_model
        nheads = self.trans_model.nhead
        
        # 通过注意力得到objects的注意力权重
        self.objects_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, hidden_dim) # FIXME


    def forward(self, encoded_feature):
        """ The forward expects a tensor from encoder, 
            in init step, we will create a mask tensor to mask padded pixels.

            It returns a BEV_object feature map 
        """
        xs = encoded_feature

        bs, c, h, w = encoded_feature.shape
        mask = torch.zeros(bs, h, w).to(encoded_feature.device)
        mask = mask != 0  # turn maks to bool
        if self.embed_method == "sine": # get position embedding from the input encoded_feature
            pos = self.position_embedding(encoded_feature, mask)  # pos 返回的是tensor， shape为[1, 256, 15, 25]
        else:
            pos = self.position_embedding(encoded_feature)
        src = self.input_proj(encoded_feature)  # project tne input encoded_feature into hidden_dim dimension [bs, hidden_dim, h, w]
        hs, memory = self.trans_model(src, mask, self.query_embed.weight, pos) # trans_model 返回 tgt 和 mask。
                                                                               # 若return_intermediate_dec为true, 则返回tgt的shape为[n, num_queries, bs, hidden_dim], n为trans_encoder和trans_decoder对应的层数
        object_mask = self.objects_attention(hs[-1], memory, mask)  # 用hs[-1]作为输入 维度为[bs, num_queries, hidden_dim]
        seg_mask = self.mask_head(src, object_mask)
        if verbose: print("in build detr, the seg_mask shape is ", seg_mask.shape)
        outputs_seg = seg_mask.view(bs, self.num_queries, seg_mask.shape[-2], seg_mask.shape[-1])   # FIXME TODO
        if verbose: print("in build detr, outputs_seg shape is ", outputs_seg.shape)
        return outputs_seg
        




if __name__ == "__main__":
    args = get_args()
    test_input = torch.randn(3, 512, 15, 25)
    # mask = torch.zeros(1, 15, 25)
    # mask = mask != 0
    # # position_embedding = build_position_encoding(hidden_dim=256, position_embedding='sine')
    # # pos = position_embedding(test_input)
    # #print("in BEV_transformer, pos of the encoded feature: ", pos.shape)
    # query_embed, input_proj, position_embedding, trans_model = build_transformer(args, num_queries=100, backbone_num_channels=512)
    # print("in BEV_transformer, transformer.d_model: ", trans_model.d_model)
    # if args.position_embedding == "sine":
    #     pos = position_embedding(test_input, mask)
    # else:
    #     pos = position_embedding(test_input)
    # print("in BEV_transformer, pos of the encoded feature: ", pos.shape)
    # output = trans_model(src=input_proj(test_input), mask=mask, query_embed=query_embed.weight, pos_embed=pos)[0]  # output 返回tgt和mask
    # #print("in BEV_transformer, output from the transformer: ", len(output))
    # print("in BEV_transformer, output from the transformer: ", output.shape)

    test_input_chnls = test_input.shape[1]
    detr = build_obj_transformer(args, test_input_chnls, 50, "sine")
    output = detr(test_input)
    print("in BEV_transformer, output shape: ", output.shape)
    print("The end")

