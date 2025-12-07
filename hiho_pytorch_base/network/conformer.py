"""CONFORMERエンコーダーの実装"""

import torch
from torch import nn
from torch.nn import functional as F


class RelativeMultiHeadAttention(nn.Module):
    """相対位置符号化付きマルチヘッド注意機構"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(  # noqa: D102
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()

        # Multi-head attention
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(out)


class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ConvolutionModule(nn.Module):
    """畳み込みモジュール"""

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, seq, dim) -> (batch, dim, seq)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (batch, dim, seq) -> (batch, seq, dim)


class ConformerBlock(nn.Module):
    """conformerブロック"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.mha = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(  # noqa: D102
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Feed forward 1
        x = x + 0.5 * self.dropout(self.ff1(self.norm1(x)))

        # Multi-head attention
        x = x + self.dropout(
            self.mha(self.norm2(x), self.norm2(x), self.norm2(x), mask)
        )

        # Convolution
        x = x + self.dropout(self.conv(self.norm3(x)))

        # Feed forward 2
        x = x + 0.5 * self.dropout(self.ff2(self.norm4(x)))

        return x


class ConformerEncoder(nn.Module):
    """conformerエンコーダー"""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(  # noqa: D102
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.layers:
            x = layer(x, mask)
        return x, mask
