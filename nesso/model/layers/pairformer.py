# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend


from typing import Optional

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from nesso.data import const
from nesso.model.layers.dropout import get_dropout_mask
from nesso.model.layers.transition import Transition
from nesso.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from nesso.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairformerNoSeqLayer(nn.Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask, use_kernels=use_kernels)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask, use_kernels=use_kernels)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )

        z = z + self.transition_z(z)
        return z


class PairformerNoSeqModule(nn.Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                PairformerNoSeqLayer(
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                ),
            )

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                z = checkpoint(
                    layer,
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels,
                    use_reentrant=False,
                )
            else:
                z = layer(
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels,
                )
        return z
