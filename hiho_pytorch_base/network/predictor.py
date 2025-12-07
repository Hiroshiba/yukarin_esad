"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


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
    ):
        super().__init__()

        self.hidden_size = hidden_size

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
        self.pre_conformer = nn.Linear(
            embedding_size + speaker_embedding_size, hidden_size
        )

        self.encoder = encoder

        # 出力ヘッド
        self.f0_head = nn.Linear(hidden_size, 1)  # F0予測用
        self.vuv_head = nn.Linear(hidden_size, 1)  # vuv予測用

    def forward(  # noqa: D102
        self,
        *,
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        speaker_id: Tensor,  # (B,)
    ) -> tuple[list[Tensor], list[Tensor]]:  # f0_list [(vL,)], vuv_list [(vL,)]
        device = speaker_id.device
        batch_size = len(phoneme_ids_list)

        # シーケンスをパディング
        phoneme_lengths = torch.tensor(
            [seq.shape[0] for seq in phoneme_ids_list], device=device
        )
        padded_phoneme_ids = pad_sequence(phoneme_ids_list, batch_first=True)  # (B, L)
        padded_durations = pad_sequence(
            phoneme_durations_list, batch_first=True
        )  # (B, L)
        padded_phoneme_stress = pad_sequence(
            phoneme_stress_list, batch_first=True
        )  # (B, L)

        # 埋め込み
        phoneme_embed = self.phoneme_embedder(padded_phoneme_ids)  # (B, L, ?)
        stress_embed = self.stress_embedder(padded_phoneme_stress)  # (B, L, ?)

        # 話者埋め込み
        speaker_embed = self.speaker_embedder(speaker_id)  # (B, ?)
        max_length = padded_phoneme_ids.size(1)
        speaker_embed = speaker_embed.unsqueeze(1).expand(
            batch_size, max_length, -1
        )  # (B, L, ?)

        # 埋め込みを結合
        h = torch.cat([phoneme_embed, stress_embed], dim=2)  # (B, L, ?)

        # 継続時間入力（オプション）
        if self.duration_linear is not None:
            duration_embed = self.duration_linear(
                padded_durations.unsqueeze(-1)
            )  # (B, L, ?)
            h = torch.cat([h, duration_embed], dim=2)

        # 話者情報を追加
        h = torch.cat([h, speaker_embed], dim=2)  # (B, L, ?)

        # Conformer前の投影
        h = self.pre_conformer(h)  # (B, L, ?)

        # マスキング
        mask = make_non_pad_mask(phoneme_lengths).unsqueeze(-2).to(device)  # (B, 1, L)

        # Conformerエンコーダ
        h, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, L, ?)

        # 出力ヘッド - 全音素に対して予測
        f0 = self.f0_head(h).squeeze(-1)  # (B, L)
        vuv = self.vuv_head(h).squeeze(-1)  # (B, L)

        # 母音位置でフィルタ
        return (
            [f0[i, vowel_indices] for i, vowel_indices in enumerate(vowel_index_list)],
            [vuv[i, vowel_indices] for i, vowel_indices in enumerate(vowel_index_list)],
        )


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
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
    )
