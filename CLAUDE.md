# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、LibriTTSなどのデータセットを使用して音素ラベルとアクセント情報（ストレス情報）から母音ごとのピッチの高さを対数F0で予測するための機械学習フレームワークです。PyTorchベースで実装されており、多話者対応のF0予測と有声/無声予測を行います。

### F0予測アプローチ

Diffusionモデル（Rectified Flow/MeanFlow）ベースの生成方式を採用：
- 音素ID列と母音/ストレスフラグを組み合わせた埋め込みをConformerエンコーダに入力
- 各時刻の文脈特徴を抽出し、全音素のF0とvuv（有声/無声）を同時生成
- 母音フラグマスクを適用して母音部分のF0とvuvのみを抽出し最終出力とする
- 学習時：F0とvuvの両方を学習し、F0のロスはvuvが有声のところのみで計算
- 学習時：子音部分の出力や、無声部分のF0はランダムノイズとして扱う
- 推論時：子音部分の出力や、無声と判定された部分のF0をランダムノイズで上書きすることでドメインシフトを防ぐ

## 主なコンポーネント

以下の主要コンポーネントがあります。
`hiho_pytorch_base`内部のモジュール同士は必ず相対インポートで参照します。

### 設定管理 (`hiho_pytorch_base/config.py`)
```python
DataFileConfig:     # ファイルパス設定
DatasetConfig:      # データセット分割設定
NetworkConfig:      # ネットワーク構造設定
ModelConfig:        # モデル設定
TrainConfig:        # 学習パラメータ設定
ProjectConfig:      # プロジェクト情報設定
```

### 学習システム (`scripts/train.py`)
- PyTorch独自実装の学習ループ
- TensorBoard/W&B統合
- torch.amp（Automatic Mixed Precision）対応
- エポックベーススケジューラー対応
- スナップショット保存・復旧機能

### データ処理 (`hiho_pytorch_base/dataset.py`)
- 遅延読み込みによるメモリ効率化
- dataclassベースの型安全なデータ構造
- train/test/eval/valid の4種類データセット対応
- pathlistファイル方式によるファイル管理
- stemベース対応付けで異なるデータタイプを自動関連付け
- 多話者学習対応（JSON形式の話者マッピング）

### ネットワーク (`hiho_pytorch_base/network/predictor.py`)
- Diffusionベースの生成 + vuv分類
- 固定長・可変長データの統一処理
- 音素エンベディング後のLinear層4層による特徴変換
- F0とvuvを同時生成

#### Conformerベースアーキテクチャ
- **入力**: 音素ID列 + 母音/ストレスフラグ列（全音素数の長さ）
- **エンコーダ**: Conformerエンコーダによる文脈特徴抽出
- **生成ヘッド**: 全音素位置のF0値とvuvを同時生成
- **出力処理**: 母音フラグマスクで母音部分の出力のみを抽出

### 推論・生成
- `hiho_pytorch_base/generator.py`: 推論ジェネレーター
- `scripts/generate.py`: 推論実行スクリプト

### テストシステム
- 自動テストデータ生成
- エンドツーエンドテスト
- 統合テスト

## 使用方法

### 学習実行
```bash
uv run -m scripts.train <config_yaml_path> <output_dir>
```

### 推論実行
```bash
uv run -m scripts.generate --model_dir <model_dir> --output_dir <output_dir> [--use_gpu] [--num_files N]
```

### データセットチェック
```bash
uv run -m scripts.check_dataset <config_yaml_path> [--trials 10]
```

### テスト実行
```bash
uv run pytest tests/ -sv
```

### 開発環境セットアップ
```bash
uv sync
```

### 静的解析とフォーマット
```bash
uv run pyright && uv run ruff check --fix && uv run ruff format
```

## 技術仕様

### 設定ファイル
- **形式**: YAML
- **管理**: Pydanticによる型安全な設定

### 主な依存関係
- **Python**: 3.12+
- **PyTorch**: 2.7.1+
- **NumPy**: 2.2.5+
- **Pydantic**: 2.11.7+
- **librosa**: 0.11.0+（音声処理）
- その他詳細は`pyproject.toml`を参照

### パッケージ管理
- **uv**による高速パッケージ管理
- **pyproject.toml**ベースの依存関係管理

### データ設計
- **ステムベース対応**: 同じサンプルのファイルは拡張子を除いて同じ名前
- **pathlist方式**: root_dirからの相対パスでファイル管理
- **データタイプ別ディレクトリ**: 各データタイプごとに独立したディレクトリ

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## フォーク時の使用方法

このフレームワークはフォークして別プロジェクト名でパッケージ化することを想定しています。

### ディレクトリ構造の維持

フォーク後も `hiho_pytorch_base/` ディレクトリ名はそのまま維持してください。
ライブラリ内部は相対インポートで実装されているため、ディレクトリ名を変更する必要はありません。

### 拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

**注意**: フォーク前からある汎用関数の関数名やdocstringは変更してはいけません。
追従するときにコンフリクトしてしまうためです。

### パッケージ名の変更方法

パッケージ名を`hiho_pytorch_base`から必ず変更します。
フォーク先で別のパッケージ名（例: `repository_name`）として配布する場合、`pyproject.toml` を以下のように変更します：

```toml
[project]
name = "repository_name"

[tool.hatch.build.targets.wheel.sources]
"hiho_pytorch_base" = "repository_name"
```

これら以外の変更は不要です。

---

@docs/設計.md
@docs/コーディング規約.md
@.claude/hiho.md
