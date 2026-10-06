# torch-wgi

WGI向け微分可能解析順投影シミュレータの研究開発リポジトリ。

## 現在の状態

**研究計画・検証仕様の策定段階です。シミュレータ、テスト、MC比較、最適化実験は未実装・未実行です。** 本リポジトリの閾値は提案された研究用受入基準であり、達成結果・規格値・臨床性能を意味しません。

## 研究の定義

同一崩壊位置でのPET応答とCompton応答を結合し、画像空間で積分する絶対計数モデルを構築します。Cone-conditioned Tube Projectorはこの結合尤度積分を低次元化する手段であり、LORとconeの幾何学的交点座標を解く手法ではありません。

- 有限ボクセルでは同じサブボクセル点で応答を掛け、次にボクセル内積分する。
- 陽電子飛程、検出効率、未検出確率、SRF、同時計数受理を明示する。
- SDF/Soft-binningの人工的緩和幅と物理的分解能を区別する。
- Fisher情報を固定投与量・固定撮像時間で最適化し、独立したMC/実測で検証する。

## ドキュメント

- [Engineering & Validation Roadmap](docs/engineering-validation-roadmap.ja.md)
- [機械可読の検証ゲート](configs/validation-gates.yaml)
- [実験記録テンプレート](configs/experiment-manifest.example.yaml)
- [研究Issue](https://github.com/ohayotaro/torch-wgi/issues)

## 開発方針

Phase 1: 数理・勾配・随伴 → Phase 2: 物理クロスバリデーション → Phase 3: 独立データでの設計改善 → Phase 4: スケール・論文・再現可能な公開。

最適化用の近似と、検証用の高精度積分/MCを分離します。実験結果はコードSHA、設定、データSHA256、環境、統計的不確実性とともに保存し、PASS / FAIL / INCONCLUSIVEを区別します。

現時点でインストール可能なPythonパッケージや実行済みベンチマークはありません。具体的なモジュール構成・環境構築・実験手順はロードマップを参照してください。OSSライセンスの選択とデータ再配布権限の確認は公開リリース前の未完了タスクです。
