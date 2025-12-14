"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from ..data.statistics import DataStatistics
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


def get_lengths(
    tensor_list: list[Tensor],  # [(L, ?)]
) -> Tensor:  # (B,)
    """テンソルリストからlengthsを取得"""
    device = tensor_list[0].device
    lengths = torch.tensor([t.shape[0] for t in tensor_list], device=device)
    return lengths


def pad_tensor_list(
    tensor_list: list[Tensor],  # [(L, ?)]
) -> Tensor:  # (B, L, ?)
    """テンソルリストをパディング"""
    batch_size = len(tensor_list)
    if batch_size == 1:
        # NOTE: ONNX化の際にpad_sequenceがエラーになるため迂回
        padded = tensor_list[0].unsqueeze(0)
    else:
        padded = pad_sequence(tensor_list, batch_first=True)
    return padded


def create_padding_mask(
    lengths: Tensor,  # (B,)
) -> Tensor:  # (B, 1, L)
    """lengthsからパディングマスクを生成"""
    mask = make_non_pad_mask(lengths).unsqueeze(-2).to(lengths.device)
    return mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        stress_embedding_size: int,
        input_phoneme_duration: bool,
        encoder: Encoder,
        statistics: DataStatistics,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if len(statistics.f0_mean) != speaker_size:
            raise ValueError(
                f"statistics speaker_size mismatch: network={speaker_size} statistics={len(statistics.f0_mean)}"
            )

        self.register_buffer("f0_mean", torch.from_numpy(statistics.f0_mean))
        self.register_buffer("f0_std", torch.from_numpy(statistics.f0_std))
        self.register_buffer("vuv_mean", torch.from_numpy(statistics.vuv_mean))
        self.register_buffer("vuv_std", torch.from_numpy(statistics.vuv_std))

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.phoneme_embedder = nn.Sequential(
            nn.Embedding(phoneme_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
        )
        self.stress_embedder = nn.Embedding(
            4, stress_embedding_size
        )  # 子音=0, 母音=1-3

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.speaker_embedder = nn.Sequential(
            nn.Embedding(speaker_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
        )

        # 継続時間写像（オプション）
        self.duration_linear = (
            nn.Linear(1, hidden_size) if input_phoneme_duration else None
        )

        # Conformer前の写像
        embedding_size = phoneme_embedding_size + stress_embedding_size
        if input_phoneme_duration:
            embedding_size += hidden_size
        additional_size = 1 + 1 + 1 + 1  # input_f0, input_vuv, t, h
        self.pre_conformer = nn.Linear(
            embedding_size + speaker_embedding_size + additional_size, hidden_size
        )

        self.encoder = encoder

        # 出力ヘッド
        self.f0_head = nn.Linear(hidden_size, 1)  # F0予測用
        self.vuv_head = nn.Linear(hidden_size, 1)  # vuv予測用

    def forward(  # noqa: D102
        self,
        *,
        padded_phoneme_ids: Tensor,  # (B, L)
        padded_phoneme_durations: Tensor,  # (B, L)
        padded_phoneme_stress: Tensor,  # (B, L)
        padded_input_f0: Tensor,  # (B, L, 1)
        padded_input_vuv: Tensor,  # (B, L, 1)
        mask: Tensor,  # (B, 1, L)
        t: Tensor,  # (B,)
        h: Tensor,  # (B,)
        speaker_id: Tensor,  # (B,)
    ) -> tuple[Tensor, Tensor]:  # (B, L, 1), (B, L, 1)
        batch_size = t.shape[0]
        max_length = padded_phoneme_ids.size(1)

        # 埋め込み
        phoneme_embed = self.phoneme_embedder(padded_phoneme_ids)  # (B, L, ?)
        stress_embed = self.stress_embedder(padded_phoneme_stress)  # (B, L, ?)

        # 話者埋め込み
        speaker_embed = self.speaker_embedder(speaker_id)  # (B, ?)
        speaker_expanded = speaker_embed.unsqueeze(1).expand(
            batch_size, max_length, -1
        )  # (B, L, ?)

        # 埋め込みを結合
        x = torch.cat([phoneme_embed, stress_embed], dim=2)  # (B, L, ?)

        # 継続時間入力（オプション）
        if self.duration_linear is not None:
            duration_embed = self.duration_linear(
                padded_phoneme_durations.unsqueeze(-1)
            )  # (B, L, ?)
            x = torch.cat([x, duration_embed], dim=2)

        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)
        h_expanded = h.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)

        x = torch.cat(
            [
                x,
                speaker_expanded,
                padded_input_f0,
                padded_input_vuv,
                t_expanded,
                h_expanded,
            ],
            dim=2,
        )  # (B, L, ?)

        # Conformer前の投影
        x = self.pre_conformer(x)  # (B, L, ?)

        # Conformerエンコーダ
        x, _ = self.encoder(x=x, cond=None, mask=mask)  # (B, L, ?)

        # 出力ヘッド - 全音素に対して予測
        f0 = self.f0_head(x)  # (B, L, 1)
        vuv = self.vuv_head(x)  # (B, L, 1)

        return f0, vuv

    def forward_list(  # noqa: D102
        self,
        *,
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        input_f0_list: list[Tensor],  # [(L,)]
        input_vuv_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        t: Tensor,  # (B,)
        h: Tensor,  # (B,)
        speaker_id: Tensor,  # (B,)
    ) -> tuple[list[Tensor], list[Tensor]]:  # [(vL,)], [(vL,)]
        lengths = get_lengths(phoneme_ids_list)  # (B,)
        padded_phoneme_ids = pad_tensor_list(phoneme_ids_list)  # (B, L)
        padded_phoneme_durations = pad_tensor_list(phoneme_durations_list)  # (B, L)
        padded_phoneme_stress = pad_tensor_list(phoneme_stress_list)  # (B, L)
        padded_input_f0 = pad_tensor_list([t.unsqueeze(-1) for t in input_f0_list])
        padded_input_vuv = pad_tensor_list([t.unsqueeze(-1) for t in input_vuv_list])
        mask = create_padding_mask(lengths)  # (B, 1, L)

        output_f0, output_vuv = self(
            padded_phoneme_ids=padded_phoneme_ids,
            padded_phoneme_durations=padded_phoneme_durations,
            padded_phoneme_stress=padded_phoneme_stress,
            padded_input_f0=padded_input_f0,
            padded_input_vuv=padded_input_vuv,
            mask=mask,
            t=t,
            h=h,
            speaker_id=speaker_id,
        )  # (B, L, 1), (B, L, 1)

        # 母音位置でフィルタ
        f0_list = [
            output_f0[i, :length, 0][vowel_index]
            for i, (length, vowel_index) in enumerate(
                zip(lengths, vowel_index_list, strict=True)
            )
        ]
        vuv_list = [
            output_vuv[i, :length, 0][vowel_index]
            for i, (length, vowel_index) in enumerate(
                zip(lengths, vowel_index_list, strict=True)
            )
        ]

        return f0_list, vuv_list


def create_predictor(config: NetworkConfig, *, statistics: DataStatistics) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=False,
        use_conv_glu_module=False,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        stress_embedding_size=config.stress_embedding_size,
        input_phoneme_duration=config.input_phoneme_duration,
        encoder=encoder,
        statistics=statistics,
    )
