# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

from torch import Tensor, nn

from nesso.data import const
from nesso.model.modules.encoders import AtomAttentionEncoder, AtomEncoder


class InputEmbedder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        use_no_atom_char: bool = False,
        add_additional_atom_features: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom embedding size.
        atom_z : int
            The atom pairwise embedding size.
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.token_s = token_s
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat
        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=use_no_atom_char,
            add_additional_atom_features=add_additional_atom_features,
        )

        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
            activation_checkpointing=activation_checkpointing,
        )

        self.res_type_encoding = nn.Linear(const.num_tokens, token_s, bias=False)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
            self.cyclic_conditioning_init.weight.data.fill_(0)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(const.chain_type_ids), token_s
            )
            self.mol_type_conditioning_init.weight.data.fill_(0)

    def forward(self, feats: dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"].float()

        # Compute input embedding
        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = self.atom_enc_proj_z(p)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )

        s = a + self.res_type_encoding(res_type)
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])

        return s
