# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import HashingMemory


N_MAX_POSITIONS = 512  # maximum input sequence length

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]


logger = getLogger()

def extract_top_level_dict(current_dict):
    if current_dict is None: return None

    output_dict = dict()
    for key in current_dict.keys():
        top_level = key.split(".")[0]
        sub_level = ".".join(key.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict

class Embedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input, params=None):

        if params is not None:
            weight = params["weight"]
        else:
            weight = self.weight

        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, input, params=None):

        if params is not None:
            weight = params["weight"]
            bias = params["bias"]
        else:
            weight = self.weight
            bias = self.bias

        return F.linear(input, weight, bias)

class LayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input, params=None):

        if params is not None:
            weight = params["weight"]
            bias = params["bias"]
        else:
            weight = self.weight
            bias = self.bias

        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, params):
        super().__init__()
        self.asm = params.asm
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim

        if params.asm is False:
            self.proj = Linear(dim, params.n_words, bias=True)
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=params.n_words,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y, get_scores=False, params=None):
        """
        Compute the loss, and optionally the scores.
        """
        assert (y == self.pad_index).sum().item() == 0

        if self.asm is False:
            if params is not None and 'proj' in params:
                scores = self.proj(x, extract_top_level_dict(params['proj'])).view(-1, self.n_words)
            else:
                scores = self.proj(x).view(-1, self.n_words)
            loss = F.cross_entropy(scores, y, reduction='mean')
        else:
            _, loss = self.proj(x, y)
            scores = self.proj.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj.log_prob(x) if self.asm else self.proj(x)


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None, params=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        if params is not None and 'q_lin' in params:
            q = shape(self.q_lin(input, extract_top_level_dict(params['q_lin'])))
        else:
            q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)


        if kv is None:
            if params is not None and 'k_lin' in params:
                k = shape(self.k_lin(input, extract_top_level_dict(params['k_lin'])))
            else:
                k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            if params is not None and 'v_lin' in params:
                v = shape(self.v_lin(input, extract_top_level_dict(params['v_lin'])))
            else:
                v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv

            if params is not None and 'k_lin' in params:
                k = shape(self.k_lin(k, extract_top_level_dict(params['k_lin'])))
            else:
                k = shape(self.k_lin(k))                                      # (bs, n_heads, qlen, dim_per_head)
            if params is not None and 'v_lin' in params:
                v = shape(self.v_lin(v, extract_top_level_dict(params['v_lin'])))
            else:
                v = shape(self.v_lin(v))                                      # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        if params is not None and 'out_lin' in params:
            context = self.out_lin(context, extract_top_level_dict(params['out_lin']))
        else:
            context = self.out_lin(context)

        return context


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu

    def forward(self, input, params=None):

        if params is not None and 'lin1' in params:
            x = self.lin1(input, extract_top_level_dict(params['lin1']))
        else:
            x = self.lin1(input)

        x = self.act(x)

        if params is not None and 'lin2' in params:
            x = self.lin2(x, extract_top_level_dict(params['lin2']))
        else:
            x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # lang specific params
        self.lang_specific_LN = params.lang_specific_LN
        self.lang_specific_FFN = params.lang_specific_FFN
        self.lang_specific_ATTN = params.lang_specific_ATTN
        self.lang_specific_ADPT = params.lang_specific_ADPT

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, 'use_lang_emb', True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = params.emb_dim       # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        if self.lang_specific_ADPT:
            self.ADPT_dim = params.ADPT_dim
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

        if self.n_langs > 1 and self.lang_specific_LN:
            self.layer_norm_emb = nn.ModuleDict()
            for lg in self.lang2id:
                self.layer_norm_emb[lg] = LayerNorm(self.dim, eps=1e-12)
        else:
            self.layer_norm_emb = LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        if self.n_langs > 1 and self.lang_specific_ATTN:
            self.attentions = nn.ModuleDict()
            for lg in self.lang2id:
                self.attentions[lg] = nn.ModuleList()
        else:
            self.attentions = nn.ModuleList()

        if self.n_langs > 1 and self.lang_specific_FFN:
            self.ffns = nn.ModuleDict()
            for lg in self.lang2id:
                self.ffns[lg] = nn.ModuleList()
        else:
            self.ffns = nn.ModuleList()

        if self.n_langs > 1 and self.lang_specific_LN:
            self.layer_norm1 = nn.ModuleDict()
            for lg in self.lang2id:
                self.layer_norm1[lg] = nn.ModuleList()
        else:
            self.layer_norm1 = nn.ModuleList()

        if self.n_langs > 1 and self.lang_specific_LN:
            self.layer_norm2 = nn.ModuleDict()
            for lg in self.lang2id:
                self.layer_norm2[lg] = nn.ModuleList()
        else:
            self.layer_norm2 = nn.ModuleList()

        if self.lang_specific_ADPT:
            self.adapter1 = nn.ModuleDict()
            for lg in self.lang2id:
                self.adapter1[lg] = nn.ModuleList()

            self.adapter2 = nn.ModuleDict()
            for lg in self.lang2id:
                self.adapter2[lg] = nn.ModuleList()

        # memories
        self.memories = nn.ModuleDict()
        if getattr(params, 'use_memory', False):
            mem_positions = params.mem_enc_positions if is_encoder else params.mem_dec_positions
            for layer_id, pos in mem_positions:
                assert 0 <= layer_id <= params.n_layers - 1
                assert pos in ['in', 'after']
                self.memories['%i_%s' % (layer_id, pos)] = HashingMemory.build(self.dim, self.dim, params)

        for layer_id in range(self.n_layers):
            if self.n_langs > 1 and self.lang_specific_ATTN:
                for lg in self.lang2id:
                    self.attentions[lg].append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            else:
                self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))

            if self.lang_specific_LN:
                for lg in self.lang2id:
                    self.layer_norm1[lg].append(LayerNorm(self.dim, eps=1e-12))
            else:
                self.layer_norm1.append(LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))

            if self.n_langs > 1 and self.lang_specific_FFN:
                for lg in self.lang2id:
                    self.ffns[lg].append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
            else:
                self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))

            if self.lang_specific_LN:
                for lg in self.lang2id:
                    self.layer_norm2[lg].append(LayerNorm(self.dim, eps=1e-12))
            else:
                self.layer_norm2.append(LayerNorm(self.dim, eps=1e-12))

            if self.lang_specific_ADPT:
                for lg in self.lang2id:
                    self.adapter1[lg].append(TransformerFFN(self.dim, self.ADPT_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
                    self.adapter2[lg].append(TransformerFFN(self.dim, self.ADPT_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None, lang=None, params=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # Second order params
        if params is not None:
            param_dict = extract_top_level_dict(params)

        # embeddings
        if params is not None and 'embeddings' in param_dict:
            tensor = self.embeddings(x, param_dict['embeddings'])
        else:
            tensor = self.embeddings(x)

        if params is not None and 'position_embeddings' in param_dict:
            tensor = tensor + self.position_embeddings(positions, param_dict['position_embeddings']).expand_as(tensor)
        else:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        if langs is not None and self.use_lang_emb:
            if params is not None and 'lang_embeddings' in param_dict:
                tensor = tensor + self.lang_embeddings(langs, param_dict['lang_embeddings'])
            else:
                tensor = tensor + self.lang_embeddings(langs)

        if self.n_langs > 1 and self.lang_specific_LN:
            assert lang is not None
            if params is not None and 'layer_norm_emb' in param_dict:
                lang_spec_layer_norm_param = extract_top_level_dict(param_dict['layer_norm_emb'])
                tensor = self.layer_norm_emb[lang](tensor, lang_spec_layer_norm_param[lang])
            else:
                tensor = self.layer_norm_emb[lang](tensor)
        else:
            if params is not None and 'layer_norm_emb' in param_dict:
                tensor = self.layer_norm_emb(tensor, param_dict['layer_norm_emb'])
            else:
                tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # For any feed in parameter
        attn_param = None
        if params is not None and 'attentions' in param_dict:
            if self.n_langs > 1 and self.lang_specific_ATTN:
                attn_param = {k: extract_top_level_dict(v) for k, v in
                             extract_top_level_dict(param_dict['attentions']).items()}
            else:
                attn_param = extract_top_level_dict(param_dict['attentions'])

        ffn_param = None
        if params is not None and 'ffns' in param_dict:
            if self.n_langs > 1 and self.lang_specific_FFN:
                ffn_param = {k: extract_top_level_dict(v) for k, v in
                             extract_top_level_dict(param_dict['ffns']).items()}
            else:
                ffn_param = extract_top_level_dict(param_dict['ffns'])

        adpt1_param = None
        if params is not None and 'adapter1' in param_dict:
            adpt1_param = {k: extract_top_level_dict(v) for k, v in
                         extract_top_level_dict(param_dict['adapter1']).items()}


        adpt2_param = None
        if params is not None and 'adapter2' in param_dict:
            adpt2_param = {k: extract_top_level_dict(v) for k, v in
                           extract_top_level_dict(param_dict['adapter2']).items()}

        ln1_param = None
        if params is not None and 'layer_norm1' in param_dict:
            if self.n_langs > 1 and self.lang_specific_LN:
                ln1_param = {k: extract_top_level_dict(v) for k, v in
                               extract_top_level_dict(param_dict['layer_norm1']).items()}
            else:
                ln1_param = extract_top_level_dict(param_dict['layer_norm1'])

        ln2_param = None
        if params is not None and 'layer_norm2' in param_dict:
            if self.n_langs > 1 and self.lang_specific_LN:
                ln2_param = {k: extract_top_level_dict(v) for k, v in
                             extract_top_level_dict(param_dict['layer_norm2']).items()}
            else:
                ln2_param = extract_top_level_dict(param_dict['layer_norm2'])

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            if self.n_langs > 1 and self.lang_specific_ATTN:
                if attn_param is not None:
                    attn = self.attentions[lang][i](tensor, attn_mask, cache=cache, params=extract_top_level_dict(attn_param[lang][str(i)]))
                else:
                    attn = self.attentions[lang][i](tensor, attn_mask, cache=cache)
            else:
                if attn_param is not None:
                    attn = self.attentions[i](tensor, attn_mask, cache=cache, params=extract_top_level_dict(attn_param[str(i)]))
                else:
                    attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            # Adapter
            if self.lang_specific_ADPT:
                if adpt1_param is not None:
                    attn = attn + self.adapter1[lang][i](attn, extract_top_level_dict(adpt1_param[lang][str(i)]))
                else:
                    attn = attn + self.adapter1[lang][i](attn)

            tensor = tensor + attn

            if self.lang_specific_LN:
                if ln1_param is not None:
                    tensor = self.layer_norm1[lang][i](tensor, extract_top_level_dict(ln1_param[lang][str(i)]))
                else:
                    tensor = self.layer_norm1[lang][i](tensor)
            else:
                if ln1_param is not None:
                    tensor = self.layer_norm1[i](tensor, extract_top_level_dict(ln1_param[str(i)]))
                else:
                    tensor = self.layer_norm1[i](tensor)

            # FFN
            if self.n_langs > 1 and self.lang_specific_FFN:
                if ffn_param is not None:
                    ffn = self.ffns[lang][i](tensor, extract_top_level_dict(ffn_param[lang][str(i)]))
                else:
                    ffn = self.ffns[lang][i](tensor)
            else:
                if ffn_param is not None:
                    ffn = self.ffns[i](tensor, extract_top_level_dict(ffn_param[str(i)]))
                else:
                    ffn = self.ffns[i](tensor)

            # Adapter
            if self.lang_specific_ADPT:
                if adpt2_param is not None:
                    ffn = ffn + self.adapter2[lang][i](ffn, extract_top_level_dict(adpt2_param[lang][str(i)]))
                else:
                    ffn = ffn + self.adapter2[lang][i](ffn)

            tensor = tensor + ffn

            if self.lang_specific_LN:
                if ln2_param is not None:
                    tensor = self.layer_norm2[lang][i](tensor, extract_top_level_dict(ln2_param[lang][str(i)]))
                else:
                    tensor = self.layer_norm2[lang][i](tensor)
            else:
                if ln2_param is not None:
                    tensor = self.layer_norm2[i](tensor, extract_top_level_dict(ln2_param[str(i)]))
                else:
                    tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y, get_scores, params=None):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)

        if params is not None:
            param_dict = extract_top_level_dict(params)

        if params is not None and 'pred_layer' in param_dict:
            scores, loss = self.pred_layer(masked_tensor, y, get_scores, params=extract_top_level_dict(param_dict['pred_layer']))
        else:
            scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss