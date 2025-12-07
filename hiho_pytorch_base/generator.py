"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .network.predictor import Predictor, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    f0: list[Tensor]  # [(vL,)]
    vuv: list[Tensor]  # [(vL,)]


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    @torch.no_grad()
    def forward(
        self,
        *,
        phoneme_ids_list: list[Tensor],  # [(L,)]
        phoneme_durations_list: list[Tensor],  # [(L,)]
        phoneme_stress_list: list[Tensor],  # [(L,)]
        vowel_index_list: list[Tensor],  # [(vL,)]
        speaker_id: TensorLike,  # (B,)
    ) -> GeneratorOutput:
        """生成経路で推論する"""

        def _convert(
            data: TensorLike,
        ) -> Tensor:
            return to_tensor(data, self.device)

        (
            f0_output_list,  # [(vL,)]
            vuv_output_list,  # [(vL,)]
        ) = self.predictor(
            phoneme_ids_list=phoneme_ids_list,
            phoneme_durations_list=phoneme_durations_list,
            phoneme_stress_list=phoneme_stress_list,
            vowel_index_list=vowel_index_list,
            speaker_id=_convert(speaker_id),
        )

        return GeneratorOutput(
            f0=f0_output_list,  # [(vL,)]
            vuv=vuv_output_list,  # [(vL,)]
        )
