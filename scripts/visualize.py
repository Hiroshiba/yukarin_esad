"""
F0予測データセットの可視化ツール

設定ファイルからDatasetCollectionを読み込み、F0・Volume・音素・ストレス情報をGradio UIで表示する。
音素区間、ストレス情報、母音F0重心を統合的に可視化し、音声再生機能も提供する。
LibriTTSデータセット対応。
"""

import argparse
import tempfile
from dataclasses import dataclass

import gradio as gr
import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from matplotlib.figure import Figure
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.data.phoneme import ArpaPhoneme
from hiho_pytorch_base.dataset import (
    Dataset,
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)


@dataclass
class DataInfo:
    """データ情報"""

    phoneme_info: str
    speaker_id: str
    audio_path: str
    details: str


@dataclass
class FigureState:
    """図の状態"""

    main_plot_fig: Figure | None = None


def get_audio_path_from_lab(lab_file_path: UPath) -> UPath:
    """.labファイルのパスから対応する音声ファイルのパスを取得"""
    stem = lab_file_path.stem

    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"無効なステム形式: {stem}")

    speaker_id = parts[0]
    chapter_id = parts[1]

    libritts_root = UPath("/tmp/datasets/LibriTTS_clean_data/LibriTTS")
    audio_path = libritts_root / "dev-clean" / speaker_id / chapter_id / f"{stem}.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    return audio_path


def extract_audio_segment(audio_path: UPath, start_time: float, end_time: float) -> str:
    """音声ファイルから指定時間範囲を切り出して一時ファイルとして保存"""
    try:
        audio, sr = librosa.load(str(audio_path), sr=None)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        requested_length = end_sample - start_sample

        if start_sample >= len(audio) or requested_length <= 0:
            raise ValueError(f"無効な時間範囲: {start_time} - {end_time}")

        actual_start = max(0, start_sample)
        actual_end = min(len(audio), end_sample)

        if actual_start < actual_end:
            audio_segment = audio[actual_start:actual_end]
        else:
            audio_segment = np.array([], dtype=audio.dtype)

        if len(audio_segment) < requested_length:
            padding_length = requested_length - len(audio_segment)
            padding = np.zeros(padding_length, dtype=audio.dtype)
            audio_segment = np.concatenate([audio_segment, padding])

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_segment, sr)

        return temp_file.name

    except Exception as e:
        print(f"音声切り出しエラー: {e}")
        raise


class VisualizationApp:
    """可視化アプリケーション"""

    def __init__(self, config_path: UPath, initial_dataset_type: DatasetType):
        self.config_path = config_path
        self.initial_dataset_type = initial_dataset_type

        self.dataset_collection = self._create_dataset()
        self.figure_state = FigureState()

    def _create_dataset(self) -> DatasetCollection:
        """データセットを作成"""
        config = Config.from_dict(yaml.safe_load(self.config_path.read_text()))
        return create_dataset(config.dataset)

    def _get_dataset_and_data(
        self, index: int, dataset_type: DatasetType
    ) -> tuple[Dataset, OutputData, LazyInputData]:
        """データセットとデータを取得する共通処理"""
        dataset = self.dataset_collection.get(dataset_type)
        output_data = dataset[index]
        lazy_data = dataset.datas[index]
        return dataset, output_data, lazy_data

    def _get_file_info(self, index: int, dataset_type: DatasetType) -> str:
        """ファイル関連の情報テキストを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        lazy_data = dataset.datas[index]

        try:
            audio_path = get_audio_path_from_lab(lazy_data.lab_path)
            audio_path_str = str(audio_path)
        except (FileNotFoundError, ValueError):
            audio_path_str = "見つからない"

        return f"""設定ファイル: {self.config_path}

F0データパス: {lazy_data.f0_path}
Volumeデータパス: {lazy_data.volume_path}
LABデータパス: {lazy_data.lab_path}
話者ID: {lazy_data.speaker_id}
音声ファイル: {audio_path_str}"""

    def _create_data_processing_text(
        self, output_data: OutputData, phonemes: list[ArpaPhoneme]
    ) -> str:
        """データ処理結果の情報テキストを作成"""
        vowel_voiced_count = output_data.vowel_voiced.sum().item()
        total_vowels = len(output_data.vowel_voiced)
        return f"""母音F0重心 shape: {tuple(output_data.vowel_f0_means.shape)}
音素ストレス shape: {tuple(output_data.phoneme_stress.shape)}
母音有声情報 shape: {tuple(output_data.vowel_voiced.shape)}

音素数: {len(phonemes)}
母音数: {total_vowels}
有声母音数: {vowel_voiced_count}
話者ID: {output_data.speaker_id.item()}"""

    def _create_integrated_f0_plot(
        self,
        output_data: OutputData,
        lazy_data: LazyInputData,
        phonemes: list[ArpaPhoneme],
        f0_rate: float,
        time_start: float,
        time_end: float,
    ) -> Figure:
        """統合F0プロットを作成"""
        input_data = lazy_data.fetch()
        f0_values = input_data.f0_data.array.flatten()
        volume_values = input_data.volume_data.array.flatten()
        vowel_f0_means = output_data.vowel_f0_means.detach().numpy()
        stress_values = output_data.phoneme_stress.detach().numpy()
        vowel_voiced = output_data.vowel_voiced.detach().numpy()

        self.figure_state.main_plot_fig, ax1 = plt.subplots(1, 1, figsize=(24, 6))

        f0_time = np.arange(len(f0_values)) / f0_rate
        volume_time = np.arange(len(volume_values)) / f0_rate
        f0_values_display = f0_values.copy()
        f0_values_display[f0_values == 0] = np.nan

        ax1.plot(f0_time, f0_values_display, "b-", linewidth=3, label="F0")
        ax1.set_xlabel("時間 (秒)", fontsize=20)
        ax1.set_ylabel("F0 (Hz)", fontsize=20, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(
            volume_time,
            volume_values,
            "r-",
            linewidth=2,
            label="Volume",
            alpha=0.7,
        )
        ax2.set_ylabel("Volume (dB)", fontsize=20, color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        ax1.set_xlim(time_start, time_end)
        ax2.set_xlim(time_start, time_end)

        y_min, y_max = ax1.get_ylim()

        cmap = plt.cm.tab20  # type: ignore
        max_phoneme_id = len(ArpaPhoneme.phoneme_list)

        for phoneme_idx, phoneme in enumerate(phonemes):
            if phoneme.end >= time_start and phoneme.start <= time_end:
                color = cmap(phoneme.phoneme_id / max_phoneme_id)
                ax1.axvspan(phoneme.start, phoneme.end, alpha=0.3, color=color)
                mid_time = (phoneme.start + phoneme.end) / 2
                if time_start <= mid_time <= time_end:
                    ax1.text(
                        mid_time,
                        y_max - (y_max - y_min) * 0.1,
                        phoneme.phoneme,
                        ha="center",
                        va="top",
                        fontsize=18,
                        rotation=0,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    )

                    if ArpaPhoneme.is_vowel(phoneme.phoneme):
                        stress_value = stress_values[phoneme_idx] - 1  # 1,2,3 -> 0,1,2
                        if stress_value > 0:
                            stress_display = "*" * stress_value
                            ax1.text(
                                mid_time,
                                y_max - (y_max - y_min) * 0.05,
                                stress_display,
                                ha="center",
                                va="top",
                                fontsize=28,
                                fontweight="bold",
                                color="black",
                            )

        vowel_idx = 0
        for phoneme in phonemes:
            if ArpaPhoneme.is_vowel(phoneme.phoneme):
                if phoneme.end >= time_start and phoneme.start <= time_end:
                    mid_time = (phoneme.start + phoneme.end) / 2
                    if vowel_idx < len(vowel_f0_means):
                        is_voiced = vowel_voiced[vowel_idx]
                        f0_value = vowel_f0_means[vowel_idx]

                        if is_voiced:
                            color = "red"
                            marker = "o"
                            bbox_color = "yellow"
                            annotation_text = f"{f0_value:.1f}Hz"
                            display_y = f0_value
                        else:
                            color = "gray"
                            marker = "s"
                            bbox_color = "lightgray"
                            annotation_text = "無声"
                            display_y = 50

                        ax1.scatter(
                            mid_time,
                            display_y,
                            s=200,
                            c=color,
                            marker=marker,
                            edgecolors="black",
                            linewidth=2,
                            zorder=10,
                            label="Vowel F0" if vowel_idx == 0 else "",
                        )

                        ax1.annotate(
                            annotation_text,
                            xy=(float(mid_time), float(display_y)),
                            xytext=(5, 10),
                            textcoords="offset points",
                            fontsize=14,
                            fontweight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor=bbox_color,
                                alpha=0.8,
                            ),
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="arc3,rad=0",
                            ),
                        )
                vowel_idx += 1

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=18,
            loc="lower right",
            ncol=len(labels1 + labels2),
        )

        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="both", which="major", labelsize=18)
        ax2.tick_params(axis="both", which="major", labelsize=18)

        plt.tight_layout()
        return self.figure_state.main_plot_fig

    def _create_plots(
        self,
        index: int,
        dataset_type: DatasetType,
        time_start: float = 0,
        time_end: float = 2,
    ) -> Figure:
        """プロットを作成"""
        dataset, output_data, lazy_data = self._get_dataset_and_data(
            index, dataset_type
        )

        input_data = lazy_data.fetch()
        phonemes = input_data.phonemes
        f0_rate = input_data.f0_data.rate

        main_plot = self._create_integrated_f0_plot(
            output_data, lazy_data, phonemes, f0_rate, time_start, time_end
        )

        return main_plot

    def _get_data_info(self, index: int, dataset_type: DatasetType) -> DataInfo:
        """データ情報を取得"""
        dataset, output_data, lazy_data = self._get_dataset_and_data(
            index, dataset_type
        )

        input_data = lazy_data.fetch()
        phonemes = input_data.phonemes

        # 音素情報
        phoneme_info_list = []
        for p in phonemes:
            info_line = f"  {p.phoneme}: {p.start:.3f}-{p.end:.3f}s"
            if ArpaPhoneme.is_vowel(p.phoneme) and p.stress is not None:
                info_line += f" [stress: {p.stress}]"
            phoneme_info_list.append(info_line)
        phoneme_info = "\n".join(phoneme_info_list)

        try:
            audio_path = get_audio_path_from_lab(lazy_data.lab_path)
            if audio_path:
                audio_path_str = str(audio_path)
            else:
                audio_path_str = "見つからない"
        except (FileNotFoundError, ValueError):
            audio_path_str = "見つからない"

        speaker_id = f"{output_data.speaker_id.item()}"

        file_info = self._get_file_info(index, dataset_type)
        data_processing_info = self._create_data_processing_text(output_data, phonemes)
        details = f"{file_info}\n\n--- データ処理結果 ---\n{data_processing_info}"

        return DataInfo(
            phoneme_info=phoneme_info,
            speaker_id=speaker_id,
            audio_path=audio_path_str,
            details=details,
        )

    def launch(self) -> None:
        """Gradio UIを起動"""
        initial_dataset = self.dataset_collection.get(self.initial_dataset_type)
        initial_max_index = len(initial_dataset) - 1

        with gr.Blocks() as demo:
            # 状態管理
            current_index = gr.State(0)
            current_dataset_type = gr.State(self.initial_dataset_type)

            # UI コンポーネント
            with gr.Row():
                dataset_type_dropdown = gr.Dropdown(
                    choices=list(DatasetType),
                    value=self.initial_dataset_type,
                    label="データセットタイプ",
                    scale=1,
                )
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=initial_max_index,
                    value=0,
                    step=1,
                    label="サンプルインデックス",
                    scale=3,
                )

            # 状態管理
            current_time_start = gr.State(0.0)
            current_time_end = gr.State(2.0)

            @gr.render(
                inputs=[
                    current_index,
                    current_dataset_type,
                    current_time_start,
                    current_time_end,
                ]
            )
            def render_content(
                index: int,
                dataset_type: DatasetType,
                time_start: float,
                time_end: float,
            ):
                # プロットとデータ情報を取得
                main_plot = self._create_plots(
                    index, dataset_type, time_start, time_end
                )
                data_info = self._get_data_info(index, dataset_type)

                # 音声取得を試みる
                try:
                    _, _, lazy_data = self._get_dataset_and_data(index, dataset_type)
                    audio_path = get_audio_path_from_lab(lazy_data.lab_path)
                    if audio_path:
                        audio_for_gradio = extract_audio_segment(
                            audio_path, time_start, time_end
                        )
                    else:
                        audio_for_gradio = None
                except Exception as e:
                    print(f"音声取得エラー: {e}")
                    audio_for_gradio = None

                with gr.Row():
                    time_start_input = gr.Number(
                        value=time_start, label="開始時間 (秒)", scale=1
                    )
                    time_end_input = gr.Number(
                        value=time_end, label="終了時間 (秒)", scale=1
                    )
                    left_btn = gr.Button("← 左へ", scale=1)
                    right_btn = gr.Button("右へ →", scale=1)

                # 時間範囲変更時の状態更新
                def update_time_range(new_start, new_end):
                    return new_start, new_end

                time_start_input.change(
                    update_time_range,
                    inputs=[time_start_input, time_end_input],
                    outputs=[current_time_start, current_time_end],
                )

                time_end_input.change(
                    update_time_range,
                    inputs=[time_start_input, time_end_input],
                    outputs=[current_time_start, current_time_end],
                )

                # 左右移動ボタンの機能
                def move_time_window(direction, current_start, current_end):
                    window_size = current_end - current_start
                    if direction == "left":
                        new_start = max(0, current_start - window_size * 1.0)
                    else:  # right
                        new_start = current_start + window_size * 1.0
                    new_end = new_start + window_size
                    return new_start, new_end, new_start, new_end

                left_btn.click(
                    lambda s, e: move_time_window("left", s, e),
                    inputs=[current_time_start, current_time_end],
                    outputs=[
                        current_time_start,
                        current_time_end,
                        time_start_input,
                        time_end_input,
                    ],
                )

                right_btn.click(
                    lambda s, e: move_time_window("right", s, e),
                    inputs=[current_time_start, current_time_end],
                    outputs=[
                        current_time_start,
                        current_time_end,
                        time_start_input,
                        time_end_input,
                    ],
                )

                with gr.Row():
                    if audio_for_gradio:
                        gr.Audio(value=audio_for_gradio, label="表示範囲の音声再生")
                    else:
                        gr.Audio(value=None, label="音声ファイルが見つかりません")

                with gr.Row():
                    gr.Plot(value=main_plot, label="F0・Volume・音素統合可視化")

                with gr.Row():
                    with gr.Column():
                        gr.Textbox(
                            value=data_info.speaker_id,
                            label="話者ID",
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Textbox(
                            value=data_info.audio_path,
                            label="音声ファイル",
                            interactive=False,
                        )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.phoneme_info,
                    label="音素区間情報",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.details,
                    label="詳細情報",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                )

            # UI操作から状態への更新
            index_slider.change(
                lambda new_index: new_index,
                inputs=[index_slider],
                outputs=[current_index],
            )

            def handle_dataset_change(new_type):
                dataset = self.dataset_collection.get(new_type)
                max_index = len(dataset) - 1
                return (
                    0,  # current_index
                    new_type,  # current_dataset_type
                    gr.update(value=0, maximum=max_index),  # スライダー
                )

            dataset_type_dropdown.change(
                handle_dataset_change,
                inputs=[dataset_type_dropdown],
                outputs=[current_index, current_dataset_type, index_slider],
            )

            # 初期化
            demo.load(
                lambda: (0, self.initial_dataset_type, 0.0, 2.0),
                outputs=[
                    current_index,
                    current_dataset_type,
                    current_time_start,
                    current_time_end,
                ],
            )

        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


def visualize(config_path: UPath, dataset_type: DatasetType) -> None:
    """指定されたデータセットをGradio UIで可視化する"""
    app = VisualizationApp(config_path, dataset_type)
    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("config_path", type=UPath, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type",
        type=DatasetType,
        default=DatasetType.TRAIN,
        help="データセットタイプ",
    )

    args = parser.parse_args()
    visualize(config_path=args.config_path, dataset_type=args.dataset_type)
