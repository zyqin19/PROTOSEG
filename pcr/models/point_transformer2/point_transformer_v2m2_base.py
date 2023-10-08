"""
Point Transformer V2 Mode (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import torch.distributed as dist


import einops
from timm.models.layers import DropPath
import pointops

from pcr.models.builder import MODELS
from pcr.models.utils import offset2batch, batch2offset
##############
from einops import rearrange, repeat
import torch.nn.functional as F
from pcr.models.prototype.sinkhorn import distributed_sinkhorn
from pcr.models.prototype.contrast import momentum_update, l2_normalize, ProjectionHead


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


@MODELS.register_module("ptv2m2")
class PointTransformerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map",
                 num_prototype = 10,
                 update_prototype = True,
                 gamma = 0.999
                 ):
        super(PointTransformerV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

        ###############################
        self.num_prototype = num_prototype     # to add
        self.update_prototype = update_prototype    # to add
        self.gamma = gamma  # to add
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, patch_embed_channels+3),
                                       requires_grad=True)

        self.proj_head = nn.Sequential(
            nn.Linear(dec_channels[0]+3, dec_channels[0]+3),
            PointBatchNorm(dec_channels[0]+3),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0]+3, dec_channels[0]+3)
        )
        self.feat_norm = nn.LayerNorm(patch_embed_channels+3)
        self.mask_norm = nn.LayerNorm(self.num_classes)


    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, data_dict, pretrain_prototype=False, use_prototype=True):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        if use_prototype:
            gt_semantic_seg = data_dict["label"]

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        # seg_logits = self.seg_head(feat)

        ####################
        c = torch.cat((feat, coord), 1)
        _c = self.proj_head(c)
        _c = self.feat_norm(_c)
        _c = self.l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        # out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feat.shape[0], h=feat.shape[2])

        if pretrain_prototype is False and use_prototype is True:
            # gt_seg = F.interpolate(gt_semantic_seg.float(), size=feat.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_semantic_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        # n, m, k = masks.shape
        # out_seg = masks.view(n, m*k)


        return out_seg


        ##########
        # n, m, k = masks.shape
        # # Step 1: Convert the class predictions to class indices
        # pred_idx = torch.argmax(out_seg, dim=-1)
        #
        # # Step 2: Compute the prototype predictions for each point
        # proto_pred = torch.zeros((n,1), dtype=torch.long, device=out_seg.device)
        # for i in range(n):
        #     # Compute the prototype prediction for this point
        #     proto_pred[i] = torch.argmax(masks[i, :, pred_idx[i]])
        #
        # # proto_pred = torch.argmax(masks[:, :, pred_idx], dim=1)
        #
        # # Map prototype indices to class-specific indices
        # class_map = torch.zeros((m, k), dtype=torch.long, device=out_seg.device)
        # for i in range(k):
        #     class_map[:, i] = torch.arange(m) + i * m
        # proto_pred = class_map.view(-1)[proto_pred]
        #
        # return out_seg, proto_pred


        # return self.seg_head(feat)


