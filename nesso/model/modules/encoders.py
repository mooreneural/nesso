# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import one_hot

import nesso.model.layers.initialize as init
from nesso.model.layers.transition import Transition
from nesso.model.modules.transformers import AtomTransformer
from nesso.model.modules.utils import LinearNoBias


class RelativePositionEncoder(Module):
    """Algorithm 3."""

    def __init__(
        self,
        token_z: int,
        r_max: int = 32,
        s_max: int = 2,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z)
        self.fix_sym_check = fix_sym_check

    def forward(self, feats):
        b_same_chain = torch.eq(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        b_same_residue = torch.eq(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        b_same_entity = torch.eq(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        d_residue = (
            feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        )

        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        d_chain = torch.where(
            (~b_same_entity) if self.fix_sym_check else b_same_chain,
            torch.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        # Note: added  | (~b_same_entity) based on observation of ProteinX manuscript
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )
        )
        return p


class PairwiseConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=token_z, hidden=transition_expansion_factor * token_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,  # Float['b n n tz'],
        token_rel_pos_feats,  # Float['b n n 3'],
    ):  # -> Float['b n n tz']:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z


def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )  # j = 2K, i = W//2, k = h * K


class AtomEncoder(Module):
    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        structure_prediction: bool = True,
        use_no_atom_char: bool = False,
        add_additional_atom_features: bool = False,
    ):
        super().__init__()

        self.embed_atom_features = Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.use_no_atom_char = use_no_atom_char
        self.add_additional_atom_features = add_additional_atom_features

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s)
            )
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z)
            )
            init.final_init_(self.z_to_p_trans[1].weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

    def forward(
        self,
        feats,
        s_trunk=None,  # Float['bm n ts'],
        z=None,  # Float['bm n n tz'],
    ):
        with torch.autocast("cuda", enabled=False):
            B, N, _ = feats["ref_pos"].shape
            atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

            atom_ref_pos = feats["ref_pos"]  # Float['b m 3'],
            atom_uid = feats["ref_space_uid"]  # Long['b m'],

            atom_feats = [atom_ref_pos, feats["ref_element"]]
            if not self.use_no_atom_char:
                atom_feats.append(feats["ref_atom_name_chars"].reshape(B, N, 4 * 64))
            if self.add_additional_atom_features:
                atom_feats.append(feats["ref_charge"].unsqueeze(-1))
                atom_feats.append(feats["ref_chirality"].unsqueeze(-1))
                atom_feats.append(feats["ref_hybridization"].unsqueeze(-1))

            atom_feats = torch.cat(atom_feats, dim=-1)

            c = self.embed_atom_features(atom_feats)

            # note we are already creating the windows to make it more efficient
            W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
            B, N = c.shape[:2]
            K = N // W
            keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
            to_keys = partial(
                single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
            )

            atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
            atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

            d = atom_ref_pos_keys - atom_ref_pos_queries  # Float['b k w h 3']
            d_norm = torch.sum(d * d, dim=-1, keepdim=True)  # Float['b k w h 1']
            d_norm = 1 / (
                1 + d_norm
            )  # AF3 feeds in the reciprocal of the distance norm

            atom_mask_queries = atom_mask.view(B, K, W, 1)
            atom_mask_keys = (
                to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
            )
            atom_uid_queries = atom_uid.view(B, K, W, 1)
            atom_uid_keys = (
                to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
            )
            v = (
                (
                    atom_mask_queries
                    & atom_mask_keys
                    & (atom_uid_queries == atom_uid_keys)
                )
                .float()
                .unsqueeze(-1)
            )  # Bool['b k w h 1']

            p = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v

            q = c

            if self.structure_prediction:
                # run only in structure model not in initial encoding
                atom_to_token = feats["atom_to_token"].float()  # Long['b m n'],

                s_to_c = self.s_to_c_trans(s_trunk.float())
                s_to_c = torch.bmm(atom_to_token, s_to_c)
                c = c + s_to_c.to(c)

                atom_to_token_queries = atom_to_token.view(
                    B, K, W, atom_to_token.shape[-1]
                )
                atom_to_token_keys = to_keys(atom_to_token)
                z_to_p = self.z_to_p_trans(z.float())
                z_to_p = torch.einsum(
                    "bijd,bwki,bwlj->bwkld",
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                )
                p = p + z_to_p.to(p)

            p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)
        return q, c, p, to_keys


class AtomAttentionEncoder(Module):
    def __init__(
        self,
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        structure_prediction: bool = True,
        activation_checkpointing: bool = False,
        transformer_post_layer_norm: bool = False,
        structure_conditioning: bool = False,
    ):
        super().__init__()

        self.structure_prediction = structure_prediction
        self.structure_conditioning = structure_conditioning
        if structure_prediction:
            input_dim = 6 if structure_conditioning else 3
            self.r_to_q_trans = LinearNoBias(input_dim, atom_s)
            init.final_init_(self.r_to_q_trans.weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, 2 * token_s if structure_prediction else token_s),
            nn.ReLU(),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        q: torch.Tensor,
        c: torch.Tensor,
        atom_enc_bias: torch.Tensor,
        to_keys: torch.Tensor,
        r: Optional[torch.Tensor] = None,  # Float['bm m 3'],
        atom_cond_coords: Optional[torch.Tensor] = None,
        multiplicity: int = 1,
    ):
        atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            # we repeat_interleave in structure_conditioning (if present)
            if q.shape[0] != r.shape[0]:
                q = q.repeat_interleave(multiplicity, 0)

            if self.structure_conditioning:
                if atom_cond_coords is None:
                    raise ValueError(
                        "atom_cond_coords must be provided if structure_conditioning is True"
                    )
                r = torch.cat([r, atom_cond_coords], dim=-1)

            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(q).float()
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        a = a.to(q)

        return a, q, c, to_keys
