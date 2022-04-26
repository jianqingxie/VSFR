from torch.nn import functional as F

import irpe
from models.rstnet.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.rstnet.attention import MultiHeadGeometryAttention, MultiHeadAttention
from models.rstnet.grid_aug import BoxRelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, rpe_config=None, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, rpe_config, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):

        # q, k = (queries + pos, keys + pos) if pos is not None else (queries, keys)
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, rpe_config=None, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, rpe_config,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # # grid geometry embedding
        # # follow implementation of https://github.com/yahoo/object_relation_transformer/blob/ec4a29904035e4b3030a9447d14c323b4f321191/models/RelationTransformerModel.py
        # relative_geometry_embeddings = BoxRelationalEmbedding(input)
        # flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        # box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        # box_size_per_head.insert(1, 1)
        # relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        # relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        # relative_geometry_weights = F.relu(relative_geometry_weights)

        out = input
        for layer in self.layers:
            out = layer(out, out, out, None, attention_mask, attention_weights, pos=pos)

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, vocab_size, word_padding_idx, d_model=512, d_in=2048, iterate_times=2, enc_rpe2d=None, use_MAA=True, **kwargs):

        if enc_rpe2d is None or len(enc_rpe2d) == 0:
            rpe_config = None
        else:
            try:
                # rpe-{ratio}-{method}-{mode}-{shared_head}-{rpe_on}
                sp = enc_rpe2d.split('-')
                assert len(sp) == 6, len(sp)
                assert sp[0] == 'rpe'
                ratio = float(sp[1])
                method = sp[2]
                mode = sp[3]
                shared_head = bool(int(sp[4]))
                rpe_on = sp[5]
                rpe_config = irpe.get_rpe_config(
                    ratio=ratio,
                    method=method,
                    mode=mode,
                    shared_head=shared_head,
                    skip=0,
                    rpe_on=rpe_on,
                )
            except:
                print("Wrong Format: RPE_HELP")
                raise

        super(TransformerEncoder, self).__init__(N, padding_idx,rpe_config, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.use_MAA = use_MAA
        self.iterate_times = iterate_times
        # image encoder
        self.image_encoder = MultiLevelEncoder(N, padding_idx, **kwargs)
        # Concept encoder
        self.concept_encoder = nn.Embedding(vocab_size, d_model, padding_idx=word_padding_idx)
        # MAA Mutual Align Attention
        self.MAA = MutualAlignEncoder(iterate_times=iterate_times)

    def forward(self, input, image_concepts=None, attention_weights=None, pos=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        visual, attention_mask = self.image_encoder(out, attention_weights=attention_weights, pos=pos)
        if self.use_MAA and image_concepts is not None:
            assert self.iterate_times > 0, "The value of iteration_times should be greater than 0"
            textual = self.concept_encoder(image_concepts.to(torch.int64)).squeeze(1)
            SGIR, _, _ = self.MAA(visual, textual)
            return SGIR, attention_mask
        return visual, attention_mask


class RefineVisualLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dropout=0.1):
        super(RefineVisualLayer, self).__init__()

        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, textual, visual):

        refine_visual = self.enc_attn(textual, visual, visual)
        refine_visual = self.layer_norm(self.dropout(refine_visual) + visual)
        refine_visual = self.pos_ffn(refine_visual)
        # refine_visual = self.layer_norm(refine_visual + residual)

        return refine_visual


class RefineTextualLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dropout=0.1):
        super(RefineTextualLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dorpout = nn.Dropout(dropout)


    def forward(self, visual, textual):

        refine_textual = self.enc_attn(visual, textual, textual)
        refine_textual = self.layer_norm(self.dorpout(refine_textual) + textual)
        refine_textual = self.pos_ffn(refine_textual)

        return refine_textual


class MutualAlignEncoder(nn.Module):


    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, dropout=.1, d_ff=2048, iterate_times=2):
        super(MutualAlignEncoder, self).__init__()
        assert d_model == h * d_k and d_k == d_v

        self.iterate_times = iterate_times

        self.refine_visual_layer = RefineVisualLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.refine_textual_layer = RefineTextualLayer(d_model, d_k, d_v, h, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, visual, textual):

        refine_visual = visual
        refine_textual = textual
        # Mutual Iterative Attention
        for i in range(self.iterate_times):
            refine_textual = self.refine_textual_layer(refine_visual, refine_textual)
            refine_visual = self.refine_visual_layer(refine_textual, refine_visual)

        SGIR = self.layer_norm(refine_visual + refine_textual)  # SGIR: Semantic-Grounded Image Representations

        return SGIR, refine_visual, refine_textual