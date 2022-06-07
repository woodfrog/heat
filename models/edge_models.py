# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from models.mlp import MLP
from models.deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from models.ops.modules import MSDeformAttn
from models.corner_models import PositionEmbeddingSine
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import torch.nn.functional as F
from utils.misc import NestedTensor


class HeatEdge(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_feature_levels, backbone_strides, backbone_num_channels, ):
        super(HeatEdge, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)

        self.edge_input_fc = nn.Linear(input_dim * 2, hidden_dim)
        self.output_fc = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim // 2, output_dim=2, num_layers=2)

        self.transformer = EdgeTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
                                           num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)

    @staticmethod
    def get_ms_feat(xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    def forward(self, image_feats, feat_mask, corner_outputs, edge_coords, edge_masks, gt_values, corner_nums,
                max_candidates, do_inference=False):
        # Prepare ConvNet features
        features = self.get_ms_feat(image_feats, feat_mask)

        srcs = []
        masks = []
        all_pos = []

        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)

        bs = edge_masks.size(0)
        num_edges = edge_masks.size(1)

        corner_feats = corner_outputs
        edge_feats = list()
        for b_i in range(bs):
            feats = corner_feats[b_i, edge_coords[b_i, :, :, 1], edge_coords[b_i, :, :, 0], :]
            edge_feats.append(feats)
        edge_feats = torch.stack(edge_feats, dim=0)
        edge_feats = edge_feats.view(bs, num_edges, -1)

        edge_inputs = self.edge_input_fc(edge_feats.view(bs * num_edges, -1))
        edge_inputs = edge_inputs.view(bs, num_edges, -1)

        edge_center = (edge_coords[:, :, 0, :].float() + edge_coords[:, :, 1, :].float()) / 2
        edge_center = edge_center / feat_mask.shape[1]

        logits_per_edge, logits_hb, logits_rel, selection_ids, s2_attn_mask, s2_gt_values = self.transformer(srcs,
                                                                                                             masks,
                                                                                                             all_pos,
                                                                                                             edge_inputs,
                                                                                                             edge_center,
                                                                                                             gt_values,
                                                                                                             edge_masks,
                                                                                                             corner_nums,
                                                                                                             max_candidates,
                                                                                                             do_inference)

        return logits_per_edge, logits_hb, logits_rel, selection_ids, s2_attn_mask, s2_gt_values


class EdgeTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 ):
        super(EdgeTransformer, self).__init__()

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        # one-layer decoder, without self-attention layers
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)

        # edge decoder w/ self-attention layers (image-aware decoder and geom-only decoder)
        self.relational_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                               return_intermediate_dec, with_sa=True)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.gt_label_embed = nn.Embedding(3, d_model)

        self.input_fc_hb = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)
        self.input_fc_rel = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)

        self.output_fc_1 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)
        self.output_fc_2 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)
        self.output_fc_3 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, reference_points, labels, key_padding_mask, corner_nums,
                max_candidates, do_inference=False):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape

        tgt = query_embed

        # per-edge filtering with single-layer decoder (no self-attn)
        hs_per_edge, _ = self.per_edge_decoder(tgt, reference_points, memory,
                                               spatial_shapes, level_start_index, valid_ratios, query_embed,
                                               mask_flatten)
        logits_per_edge = self.output_fc_1(hs_per_edge).permute(0, 2, 1)
        filtered_hs, filtered_mask, filtered_query, filtered_rp, filtered_labels, selected_ids = self.candidate_filtering(
            logits_per_edge,
            hs_per_edge, query_embed, reference_points,
            labels,
            key_padding_mask, corner_nums, max_candidates)
        
        # generate the info for masked training
        if not do_inference:
            filtered_gt_values = self.generate_gt_masking(filtered_labels, filtered_mask)
        else:
            filtered_gt_values = filtered_labels
        gt_info = self.gt_label_embed(filtered_gt_values)

        # relational decoder with image feature (image-aware decoder)
        hybrid_prim_hs = self.input_fc_hb(torch.cat([filtered_hs, gt_info], dim=-1))

        hs, inter_references = self.relational_decoder(hybrid_prim_hs, filtered_rp, memory,
                                                       spatial_shapes, level_start_index, valid_ratios, filtered_query,
                                                       mask_flatten,
                                                       key_padding_mask=filtered_mask, get_image_feat=True)

        logits_final_hb = self.output_fc_2(hs).permute(0, 2, 1)

        # relational decoder without image feature (geom-only decoder)
        rel_prim_hs = self.input_fc_rel(torch.cat([filtered_query, gt_info], dim=-1))

        hs_rel, _ = self.relational_decoder(rel_prim_hs, filtered_rp, memory,
                                            spatial_shapes, level_start_index, valid_ratios, filtered_query,
                                            mask_flatten,
                                            key_padding_mask=filtered_mask, get_image_feat=False)

        logits_final_rel = self.output_fc_3(hs_rel).permute(0, 2, 1)

        return logits_per_edge, logits_final_hb, logits_final_rel, selected_ids, filtered_mask, filtered_gt_values

    @staticmethod
    def candidate_filtering(logits, hs, query, rp, labels, key_padding_mask, corner_nums, max_candidates):
        """
            Filter out the easy-negatives from the edge candidates, and update the edge information correspondingly
        """
        B, L, _ = hs.shape
        preds = logits.detach().softmax(1)[:, 1, :]  # BxL
        preds[key_padding_mask == True] = -1  # ignore the masking parts
        sorted_ids = torch.argsort(preds, dim=-1, descending=True)
        filtered_hs = list()
        filtered_mask = list()
        filtered_query = list()
        filtered_rp = list()
        filtered_labels = list()
        selected_ids = list()
        for b_i in range(B):
            num_candidates = corner_nums[b_i] * 3
            ids = sorted_ids[b_i, :max_candidates[b_i]]
            filtered_hs.append(hs[b_i][ids])
            new_mask = key_padding_mask[b_i][ids]
            new_mask[num_candidates:] = True
            filtered_mask.append(new_mask)
            filtered_query.append(query[b_i][ids])
            filtered_rp.append(rp[b_i][ids])
            filtered_labels.append(labels[b_i][ids])
            selected_ids.append(ids)
        filtered_hs = torch.stack(filtered_hs, dim=0)
        filtered_mask = torch.stack(filtered_mask, dim=0)
        filtered_query = torch.stack(filtered_query, dim=0)
        filtered_rp = torch.stack(filtered_rp, dim=0)
        filtered_labels = torch.stack(filtered_labels, dim=0)
        selected_ids = torch.stack(selected_ids, dim=0)

        return filtered_hs, filtered_mask, filtered_query, filtered_rp, filtered_labels, selected_ids

    @staticmethod
    def generate_gt_masking(labels, mask):
        """
            Generate the info for masked training on-the-fly with ratio=0.5
        """
        bs = labels.shape[0]
        gt_values = torch.zeros_like(mask).long()
        for b_i in range(bs):
            edge_length = (mask[b_i] == 0).sum()
            rand_ratio = np.random.rand() * 0.5 + 0.5
            gt_rand = torch.rand(edge_length)
            gt_flag = torch.zeros(edge_length)
            gt_flag[torch.where(gt_rand >= rand_ratio)] = 1
            gt_idx = torch.where(gt_flag == 1)
            pred_idx = torch.where(gt_flag == 0)
            gt_values[b_i, gt_idx[0]] = labels[b_i, gt_idx[0]]
            gt_values[b_i, pred_idx[0]] = 2  # use 2 to represent unknown value, need to predict
        return gt_values
