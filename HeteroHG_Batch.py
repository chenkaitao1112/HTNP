import os
import warnings
from typing import List, Tuple
import torch.nn.functional as F
import dhg
import torch
import torch.nn as nn
import numpy as np
import pickle
from config import args
device = torch.device('cuda:{}'.format(args.gpu))



class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)  #

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x).to(device)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)

class TimeEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, period, dropout=0.1):
        super(TimeEmbedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)
        self.period = period
        self.sincos_proj = nn.Linear(2, d_model)
        self.fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, x):

        B, L = x.size()
        pos = torch.arange(L, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # [B, L]
        tok_emb = self.tok_embed(x) + self.pos_embed(pos)  # [B, L, d_model]
        x_float = x.float()
        sincos = torch.stack([
            torch.sin(2 * np.pi * x_float / self.period),
            torch.cos(2 * np.pi * x_float / self.period)
        ], dim=-1)
        sincos_emb = self.sincos_proj(sincos)  # [B, L, d_model]
        fused = torch.cat([tok_emb, sincos_emb], dim=-1)  # [B, L, 2*d_model]
        out = self.fuse(fused)  # [B, L, d_model]

        return self.norm(self.dropout(out))


class Encoder_HeteroHG_transformer_batch(nn.Module):
    def __init__(self, eventlog, d_model, f):


        super(Encoder_HeteroHG_transformer_batch, self).__init__()
        dict_cat_view = {}
        dict_cat_emb = nn.ModuleDict()

        self.eventlog = eventlog
        self.d_model = d_model

        base_dir = "data"
        CE_Path = os.path.join(base_dir, eventlog, f"{eventlog}_y_CE_hyperedge.pickle")

        with open(CE_Path, "rb") as ce_file:
            self.CE_hyperedge = pickle.load(ce_file)
        with open("data/" + eventlog + "/" + eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            self.num_view = pickle.load(pickle_file)
        with open("data/" + eventlog + "/" + eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            self.cat_view = pickle.load(pickle_file)
        with open("data/" + eventlog + "/" + eventlog + '_seq_length.pickle', 'rb') as pickle_file:
            self.seq_length = pickle.load(pickle_file)
        self.seq_length = 10

        for c in self.cat_view:
            voca_size = np.load("data/" + eventlog + "/" + eventlog + '_' + c + '_' + str(f) + "_info.npy")
            if c == "activity":
                self.activity_voca_size = int(voca_size +1)
                dict_cat_view[c] = [voca_size + 2, d_model]
                dict_cat_emb[c] = Embedding(voca_size + 2, self.seq_length+1, d_model).to(device)
            elif c =="month":
                dict_cat_view[c] = [13, d_model]
                dict_cat_emb[c] = TimeEmbedding(13, self.seq_length, d_model,13).to(device)
            elif c =="weekday":
                dict_cat_view[c] = [7, d_model]
                dict_cat_emb[c] = TimeEmbedding(7, self.seq_length, d_model, 7 ).to(device)
            elif c =="day":
                dict_cat_view[c] = [32, d_model]
                dict_cat_emb[c] = TimeEmbedding(32, self.seq_length, d_model, 32).to(device)
            elif c=="hour":
                dict_cat_view[c] = [24, d_model]
                dict_cat_emb[c] = TimeEmbedding(24, self.seq_length, d_model, 24).to(device)
            else:
                clean_key = c.replace(".", "_")
                if clean_key == "type":
                    clean_key = "type_view"
                dict_cat_view[clean_key] = [voca_size + 1, d_model]
                dict_cat_emb[clean_key] = Embedding(voca_size + 1, self.seq_length, d_model).to(device)


        self.dict_cat_view = dict_cat_view
        self.dict_cat_emb = dict_cat_emb

        self.n_views = len(self.cat_view)
        self.d_model_new = d_model * (self.n_views )

        self.Prepare_batch_HG =Prepare_batch_HG(
            input_dim=d_model,
            hidden_dim=d_model,
            d_model=self.d_model,
            n_view=self.n_views,
            num_layers=2,
            dropout=0.1
        ).to(device)

        self.final_mlp = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        self.hg_processor = HeteroHypergraphProcessor(d_model, d_model, self.n_views)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.3, batch_first=True, device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.time_pre = MultiTaskHead(d_model)

        self.numerical_feature_projector = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        ).to(device)

    def forward(self, att_str, att):

        att_dict = dict(zip(att_str, att))

        reference_tensor_for_length_calc = None
        view_dict = {}
        for c in self.cat_view:
            clean = c.replace(".", "_")
            if clean == "type":
                clean = "type_view"

            embedding = self.dict_cat_emb[clean]
            input_tensor = att_dict.get(c)   #[batch_size,seq_num]

            batch_size, seq_num = input_tensor.shape
            if c != "activity":
                reference_tensor_for_length_calc = input_tensor
            view_matrix = embedding(input_tensor)  #[batch_size,seq_num,d_model]
            if c == "activity":
                view_matrix = view_matrix[:, 1:, :]
                ce_hyperedge_list = remap_hyperedges_for_batch(input_tensor, self.CE_hyperedge)
            view_dict[c] = view_matrix

        seq_lengths = torch.sum(reference_tensor_for_length_calc != 0, dim=1)
        seq_lengths = seq_lengths.to(device)
        seq_lengths_for_packing = torch.clamp(seq_lengths.clone(), min=1)

        Batch_node_num,Batch_node_feature,Batch_hyperlist,Batch_edge_type = self.Prepare_batch_HG(view_dict, seq_lengths_for_packing , ce_hyperedge_list)
        merged_node_feature, merged_hyperedge, merged_edge_type, edge_batch_vec, node_offsets, num_nodes_total = merge_batch_hypergraphs(Batch_node_num,Batch_node_feature,Batch_hyperlist,Batch_edge_type,device)

        EDGE_TYPE_WEIGHT = {0: 1.5, 1: 1.2, 2: 0.8}
        e_weight = generate_edge_weights(merged_edge_type, EDGE_TYPE_WEIGHT)
        Batch_hypergraph = dhg.Hypergraph(num_nodes_total,merged_hyperedge, e_weight=e_weight)
        e,_ = Batch_hypergraph.e

        updata_feature = self.hg_processor(merged_node_feature, Batch_hypergraph,merged_edge_type,edge_batch_vec ,node_offsets)

        batch_size = len(Batch_node_num)
        split_features = torch.split(updata_feature, Batch_node_num, dim=0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features_with_cls = []
        sequence_lengths_with_cls = []

        for i in range(batch_size):
            features_with_cls.append(torch.cat((cls_tokens[i], split_features[i]), dim=0))
            sequence_lengths_with_cls.append(features_with_cls[-1].shape[0])

        padded_transformer_input = torch.nn.utils.rnn.pad_sequence(
            features_with_cls, batch_first=True, padding_value=0.0
        )

        max_len = padded_transformer_input.size(1)
        seq_lengths_tensor = torch.tensor(sequence_lengths_with_cls, device=device)
        padding_mask = torch.arange(max_len, device=device)[None, :] >= seq_lengths_tensor[:, None]

        transformer_output = self.transformer_encoder(
            padded_transformer_input,
            src_key_padding_mask=padding_mask
        )

        cls_output = transformer_output[:, 0, :]
        final_feat = self.final_mlp(cls_output)

        activity_outputs_list = []
        for i in range(batch_size):
            num_activity = int(seq_lengths[i].item())
            sample_activity_output = transformer_output[i, 1: 1 + num_activity, :]
            activity_outputs_list.append(sample_activity_output)

        time_out_list = [self.time_pre(out) for out in activity_outputs_list]
        padded_time_out = torch.stack(time_out_list, dim=0)

        return final_feat, padded_time_out

class Prepare_batch_HG(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model ,n_view,  num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.linaer_list = nn.ModuleList().to(device)

        for i in range((n_view - 10) // 2 + 1):
            self.linaer_list.append(nn.Linear(2 * d_model, d_model).to(device))

        self.high_time_mlp = nn.Linear(3 * d_model, d_model).to(device)
        self.low_time_mlp = nn.Linear(4 * d_model, d_model).to(device)

        self.view_mlp_list = nn.ModuleList().to(device)

        for i in range((n_view - 10) // 2 + 6):
            self.view_mlp_list.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model).to(device)
                ).to(device)
            )

    def forward(self, view_dict, seq_lengths, batch_CE):
        batch_size, seq_len, _ = view_dict["activity"].shape
        Batch_node_feature = []
        Batch_hyperlist = []
        Batch_edge_type = []
        Batch_node_num = []
        for b in range(batch_size):
            seq_lengths_n = seq_lengths[b].item()
            sample_dict = {}
            sample_list = []
            rows = view_dict["activity"][b].shape[0]
            start_idx = max(0, rows - seq_lengths_n)
            sample_list.append(view_dict["activity"][b][start_idx:, :])
            num_activity = seq_lengths_n
            for key, value in view_dict.items():
                sample_dict[key] = value[b][start_idx:, :]
                if key != "activity":
                    sample_list.append(sample_dict[key])

            exclude_views = ["activity", "resource", "timestamp"]
            high_time_views = ['hour', 'minute', 'second']
            low_time_views = ['year', 'month', 'day', 'weekday']

            all_views = [key for key in sample_dict.keys() if key not in exclude_views]

            existing_high_time_views = [view for view in high_time_views if view in all_views]
            existing_low_time_views = [view for view in low_time_views if view in all_views]
            other_views = [view for view in all_views if
                           view not in high_time_views and view not in low_time_views]


            existing_high_time_datas = [sample_dict[view] for view in existing_high_time_views]
            existing_low_time_datas = [sample_dict[view] for view in existing_low_time_views]

            other_view_datas = [sample_dict[view] for view in other_views]

            merge_sample_list = []
            merge_sample_dict = {}


            for core_key in exclude_views:
                if core_key in sample_dict:
                    if isinstance(sample_dict[core_key], torch.Tensor):
                        merge_sample_list.append(sample_dict[core_key])
                        merge_sample_dict[core_key] = sample_dict[core_key]
                    else:
                        warnings.warn(f"The core view '{core_key}' is not a tensor type and has been skipped")

            high_time_merged_data = existing_high_time_datas[0]
            for data in existing_high_time_datas[1:]:
                high_time_merged_data = torch.cat([high_time_merged_data, data], dim=1)

            low_time_merged_data = existing_low_time_datas[0]
            for data in existing_low_time_datas[1:]:
                low_time_merged_data = torch.cat([low_time_merged_data, data], dim=1)

            high_time_merged_data = self.high_time_mlp(high_time_merged_data)
            high_time_merged_name = "high_time"
            low_time_merged_data = self.low_time_mlp(low_time_merged_data)
            low_time_merged_name = "low_time"

            merge_sample_list.append(high_time_merged_data)
            merge_sample_dict[high_time_merged_name] = high_time_merged_data
            merge_sample_list.append(low_time_merged_data)
            merge_sample_dict[low_time_merged_name] = low_time_merged_data
            next_linear_idx = 0

            for i in range(0, len(other_view_datas), 2):
                if not isinstance(other_view_datas[i], torch.Tensor):
                    warnings.warn(f"View '{other_views[i]}' is not a tensor type and has been skipped")
                    continue


                if i + 1 < len(other_view_datas):
                    if not isinstance(other_view_datas[i + 1], torch.Tensor):
                        warnings.warn(f"View '{other_views[i + 1]}' is not a tensor type; the current view has been processed separately")
                        merge_sample_list.append(other_view_datas[i])
                        merge_sample_dict[other_views[i]] = other_view_datas[i]
                        continue
                    merged_data = torch.cat([other_view_datas[i], other_view_datas[i + 1]], dim=1)
                    merged_data = self.linaer_list[next_linear_idx](merged_data)
                    next_linear_idx += 1

                    merged_name = f"{other_views[i]}_{other_views[i + 1]}"
                    merge_sample_list.append(merged_data)
                    merge_sample_dict[merged_name] = merged_data
                else:
                    merge_sample_list.append(other_view_datas[i])
                    merge_sample_dict[other_views[i]] = other_view_datas[i]

            for idx, key in enumerate(merge_sample_dict.keys(), start=0):
                merge_sample_dict[key] = self.view_mlp_list[idx](merge_sample_dict[key])

            Causal_hyperedge = batch_CE[b]
            new_sample_dict = {}
            new_sample_relation = {}
            new_sample_dict["resource"], new_sample_relation["resource"] = map_activity_to_other(
                merge_sample_dict["activity"], merge_sample_dict["resource"])
            num_res = new_sample_dict["resource"].shape[0]
            for key, value in merge_sample_dict.items():
                if key != "activity" and key != "resource":
                    new_sample_dict[key], new_sample_relation[key] = map_activity_to_other(
                        merge_sample_dict["activity"],
                        merge_sample_dict[key])

            node_feature, att2act_hyperedge, res_start_idx = build_activity_attribute_hyperedges_vertical(
                merge_sample_dict,
                new_sample_dict,
                new_sample_relation)
            res_knn_hyperedge = build_resource_knn_hyperedges(new_sample_dict["resource"], res_start_idx)
            num_node = node_feature.shape[0]
            HyperEdge, edge_type = merge_hyperedges_with_types3(Causal_hyperedge, att2act_hyperedge, res_knn_hyperedge)
            Batch_node_num.append(num_node)
            Batch_node_feature.append(node_feature)
            Batch_hyperlist.append(HyperEdge)
            Batch_edge_type.append(edge_type)
        return  Batch_node_num,Batch_node_feature,Batch_hyperlist,Batch_edge_type

#merage big batch hypergraph
def merge_batch_hypergraphs(
        Batch_node_num: List[int],
        Batch_node_feature: List[torch.Tensor],
        Batch_hyperlist: List[List],
        Batch_edge_type: List,
        device: torch.device
) -> Tuple[torch.Tensor, List[List[int]], torch.LongTensor, torch.LongTensor, List[int], int]:  ### 修改返回类型

    batch_size = len(Batch_node_num)
    node_offsets = []
    cur = 0
    for n in Batch_node_num:
        node_offsets.append(cur)
        cur += int(n)
    num_nodes_total = cur
    merged_node_feature = torch.cat(Batch_node_feature, dim=0).to(device)
    merged_hyperedge = []
    merged_edge_type_list = []
    edge_batch_vec_list = []

    for b in range(batch_size):
        offset = node_offsets[b]
        hyperlist_b = Batch_hyperlist[b]
        edge_type_b = Batch_edge_type[b]

        if isinstance(edge_type_b, torch.Tensor):
            edge_type_b_list = edge_type_b.detach().cpu().tolist()
        else:
            edge_type_b_list = list(edge_type_b)

        if len(edge_type_b_list) != len(hyperlist_b):
            raise ValueError(f"sample {b}: len(edge_type)={len(edge_type_b_list)} != len(hyperlist)={len(hyperlist_b)}")


        for he_idx, he in enumerate(hyperlist_b):
            he_tensor = torch.as_tensor(he, dtype=torch.long)
            if offset != 0:
                he_tensor = he_tensor + offset
            merged_hyperedge.append(he_tensor.tolist())
            merged_edge_type_list.append(int(edge_type_b_list[he_idx]))
            edge_batch_vec_list.append(b)

    if len(merged_edge_type_list) > 0:
        merged_edge_type = torch.tensor(merged_edge_type_list, dtype=torch.long, device=device)
        edge_batch_vec = torch.tensor(edge_batch_vec_list, dtype=torch.long, device=device)
    else:
        merged_edge_type = torch.empty((0,), dtype=torch.long, device=device)
        edge_batch_vec = torch.empty((0,), dtype=torch.long, device=device)

    return merged_node_feature, merged_hyperedge, merged_edge_type, edge_batch_vec, node_offsets, num_nodes_total



class HeteroHypergraphProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_view, num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.hgnn_layers = nn.ModuleList()
        self.hgnn_layers.append(HeteroHypergraphAttentionConv(input_dim, hidden_dim,3))
        for _ in range(num_layers - 2):
            self.hgnn_layers.append(HeteroHypergraphAttentionConv(hidden_dim, hidden_dim,3))
        if num_layers > 1:
            self.hgnn_layers.append(HeteroHypergraphAttentionConv(hidden_dim, hidden_dim,3))

        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)


    def forward(self, x, hg,edge_type, edge_batch_vec,node_offsets):


        residual = x
        h = x
        h =h.to(device)
        hg = hg.to(device)
        for i, layer in enumerate(self.hgnn_layers):
            h_temp = h
            h_new = layer(h, hg, edge_type, edge_batch_vec,node_offsets)
            h = h_new + h_temp

            if i < len(self.hgnn_layers) - 1:
                h = self.LayerNorm(h)
                h = F.relu(h)

        h = self.LayerNorm2(h)
        return h



class HeteroHypergraphAttentionConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_edge_types, attn_hidden_dim=None, dropout=0.1):
        super().__init__()
        if attn_hidden_dim is None:
            attn_hidden_dim = max(out_dim // 2, 1)

        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_mlps = nn.ModuleDict()
        for t in range(num_edge_types):
            self.attn_mlps[str(t)] = nn.Sequential(
                nn.Linear(out_dim, attn_hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(attn_hidden_dim, 1)
            )

    def forward(self, x: torch.Tensor, hg: dhg.Hypergraph,
                edge_type: torch.Tensor, edge_batch_vec: torch.Tensor,node_offsets):

        x_proj = self.lin(x)
        edge_feats = hg.v2e(x_proj, aggr='mean')
        attn_weights = torch.zeros(edge_feats.size(0), device=x.device)
        for type_str, mlp in self.attn_mlps.items():
            t = int(type_str)
            mask = (edge_type == t)
            if not mask.any():
                continue

            edge_feats_t = edge_feats[mask]
            edge_batch_vec_t = edge_batch_vec[mask]
            scores_t = mlp(edge_feats_t).squeeze(-1)
            unique_groups, inverse = torch.unique(edge_batch_vec_t, return_inverse=True)
            assert scores_t.numel() == edge_batch_vec_t.numel(), \
                f"scores_t={scores_t.shape}, edge_batch_vec_t={edge_batch_vec_t.shape}"
            attn_scores_t = group_softmax_cpu_gpu(scores_t, inverse.to(scores_t.device))
            attn_weights[mask] = attn_scores_t

        weighted_edge_feats = edge_feats * attn_weights.unsqueeze(1)
        x_out = hg.e2v(weighted_edge_feats, aggr='sum')  # [N_total, out_dim]


        return self.dropout(x_out)



def remap_hyperedges_for_batch(activity_tensor: torch.Tensor,
                               CE_hyperedge: list[list[int]],
                               padding_id: int = 0) -> list[list[list[int]]]:

    activity_tensor_cpu = activity_tensor.cpu()
    batch_size = activity_tensor_cpu.shape[0]
    batch_remapped_hyperedges = []
    for i in range(batch_size):
        sample_sequence = activity_tensor_cpu[i]
        sequence_list_no_padding = [token.item() for token in sample_sequence if token.item() != padding_id]
        value_to_index_map = {val: idx for idx, val in enumerate(sequence_list_no_padding)}

        sample_local_hyperedges = []
        for global_edge in CE_hyperedge:
            if all(node in value_to_index_map for node in global_edge):
                remapped_edge = [value_to_index_map[node] for node in global_edge]
                sample_local_hyperedges.append(remapped_edge)
        batch_remapped_hyperedges.append(sample_local_hyperedges)

    return batch_remapped_hyperedges




def map_activity_to_other(activity_view: torch.Tensor,
                          other_view: torch.Tensor,
                          decimal: int = 6):

    device = other_view.device
    activity_view = activity_view.to(device)
    other_view = other_view.to(device)

    with torch.no_grad():
        scale = 10 ** decimal
        rounded = torch.round(other_view * scale) / scale  # [N, D]
        _, inverse_idx = torch.unique(rounded, dim=0, return_inverse=True)
        K = int(inverse_idx.max().item()) + 1

    N, D = other_view.shape
    unique_sum = torch.zeros(K, D, device=device, dtype=other_view.dtype)
    unique_sum.index_add_(0, inverse_idx, other_view)
    counts = torch.bincount(inverse_idx, minlength=K).clamp_min(1).to(other_view.dtype).unsqueeze(1)
    unique_other_view = unique_sum / counts
    activity_relationship = [[int(i)] for i in inverse_idx.tolist()]

    return unique_other_view, activity_relationship

def build_activity_attribute_hyperedges_vertical(
    sample_dict,
    new_sample_dict,
    new_sample_relation,
    activity_col_name="activity",
    resource_col_name="resource"
):

    activity_view = sample_dict[activity_col_name]   # Tensor
    activity_sample_num = activity_view.size(0)
    view_info = []
    current_offset = 0

    view_info.append({
        "name": activity_col_name,
        "sample_num": activity_sample_num,
        "start_row": current_offset
    })
    current_offset += activity_sample_num

    attribute_view_names = [name for name in new_sample_dict.keys() if name != activity_col_name]
    for view_name in attribute_view_names:
        attr_view = new_sample_dict[view_name]  # Tensor
        attr_sample_num = attr_view.size(0)
        view_info.append({
            "name": view_name,
            "sample_num": attr_sample_num,
            "start_row": current_offset
        })
        current_offset += attr_sample_num

    try:
        resource_meta = next(v for v in view_info if v["name"] == resource_col_name)
        resource_start_idx = resource_meta["start_row"]
    except StopIteration:
        raise ValueError(f"no resource view '{resource_col_name}'")

    for v in attribute_view_names:
        new_sample_dict[v] = new_sample_dict[v].to(device)
    concat_list = [activity_view] + [new_sample_dict[v] for v in attribute_view_names]
    concat_matrix = torch.cat(concat_list, dim=0)

    activity_hyperedges = []
    total_activities = activity_sample_num

    for activity_idx in range(total_activities):
        activity_full_idx = view_info[0]["start_row"] + activity_idx
        activity_indices = [activity_full_idx]

        for view_name in attribute_view_names:
            view_meta = next(v for v in view_info if v["name"] == view_name)
            attr_start_row = view_meta["start_row"]
            attr_unique_idx = new_sample_relation[view_name][activity_idx][0]
            attr_full_idx = attr_start_row + attr_unique_idx
            activity_indices.append(attr_full_idx)

        activity_hyperedges.append(activity_indices)

    return concat_matrix, activity_hyperedges, resource_start_idx


def build_resource_knn_hyperedges(
    resource_features: torch.Tensor,
    resource_start_idx: int,
    k: int = 3
):

    num_resources = resource_features.size(0)


    if num_resources < k+2:
        return []

    k = min(k, num_resources)
    normed = torch.nn.functional.normalize(resource_features, p=2, dim=1)
    sim_matrix = torch.matmul(normed, normed.t())
    _, knn_indices = torch.topk(sim_matrix, k=k, dim=1)

    resource_hyperedges = (knn_indices + resource_start_idx).tolist()
    unique_hyperedges = []
    seen = set()
    for edge in resource_hyperedges:
        sorted_edge = sorted(edge)
        edge_tuple = tuple(sorted_edge)
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            unique_hyperedges.append(sorted_edge)

    return unique_hyperedges

def merge_hyperedges_with_types3(
        causal_hyperedge,
        same_res_hyperedge,
        att2act_hyperedge
):
    CAUSAL_TYPE = 0
    SAME_RES_TYPE = 1
    ATT2ACT_TYPE = 2

    merged_hyperedges = []
    hyperedge_types = []

    merged_hyperedges.extend(causal_hyperedge)
    hyperedge_types.extend([CAUSAL_TYPE] * len(causal_hyperedge))

    merged_hyperedges.extend(same_res_hyperedge)
    hyperedge_types.extend([SAME_RES_TYPE] * len(same_res_hyperedge))

    merged_hyperedges.extend(att2act_hyperedge)
    hyperedge_types.extend([ATT2ACT_TYPE] * len(att2act_hyperedge))

    hyperedge_types = torch.tensor(hyperedge_types, dtype=torch.long)

    return merged_hyperedges, hyperedge_types

def generate_edge_weights(edge_type: torch.Tensor, weight_map: dict) -> list:

    if edge_type.is_cuda:
        edge_type_np = edge_type.cpu().numpy()
    else:
        edge_type_np = edge_type.numpy()

    e_weight = []
    for et in edge_type_np:
        if et not in weight_map:
            raise ValueError(f"Unknown hyperedge type '{et}'; the weight map only contains {list(weight_map.keys())}")
        e_weight.append(weight_map[et])

    return e_weight


class MultiTaskHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.time_gru = nn.GRU(d_model, d_model, num_layers=1,
                               batch_first=True, bidirectional=False)
        self.time_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, gru_out):
        gru_out = gru_out.unsqueeze(0)
        out_time, _ = self.time_gru(gru_out)
        time_feat = out_time.mean(dim=1)
        time_pred = self.time_fc(time_feat).squeeze(-1)

        return  time_pred


def group_softmax_cpu_gpu(scores, group_idx):
    device = scores.device
    group_idx = group_idx.to(device)
    unique_groups = torch.unique(group_idx)
    out = torch.empty_like(scores)

    for g in unique_groups:
        mask = (group_idx == g)
        s = scores[mask]
        out[mask] = torch.softmax(s, dim=0)
    return out

