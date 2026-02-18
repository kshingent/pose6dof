# pose6dof

`pose6dof` は、（3次元剛体変換）の計算を「直感的」かつ「高効率」に行うための Python ライブラリです。SciPy の `Rotation` を基盤としつつ、ロボティクスにおけるリー群論（Lie Group Theory）とねじ理論（Screw Theory）の実践的な実装を統合しています。

---

## コア・コンセプト

### ハイブリッド評価システム

計算リソースとメモリのトレードオフを最適化するため、**「遅延評価（Lazy Evaluation）」** と **「キャッシュ（Caching）」** を組み合わせています。

* **Lazy Evaluation**: 指数写像やオイラー角など、計算負荷の高い表現はプロパティにアクセスされるまで計算を実行しません。
* **Caching**: 一度計算された表現は内部に保持され、2回目以降のアクセスは完全な O(1) です。
* **Eager Mode**: 初期化時に `eager=True` を指定することで、全表現を事前に計算（フロントロード）し、リアルタイム処理中のジッターを排除できます。

---

## 主要機能

### 1. 入力自動判定 (Auto-Detection)

コンストラクタは、引数の形状（Shape）や型（Type）から入力を自動的に判別します。

| 入力形式 | 判断基準 | 内部処理 |
| --- | --- | --- |
| **Affine Matrix** | 4×4行列 | 同次変換行列として読み込み。回転行列部分の直交性をチェック。 |
| **Exponential Coordinates** | 6次元ベクトル | se(3)リー代数として処理。指数写像を適用。 |
| **Separated (pos, rot)** | 引数が2つ、または名前付き引数 | 並進と回転を個別に処理。 |

**回転（rot）のサポート形式:**

* 3×3行列 (Rotation Matrix)
* 長さ 4 の配列 (Quaternion: w, x, y, z)
* 長さ 3 の配列 (Rotation Vector / Axis-angle)
* SciPy `Rotation` オブジェクト
* 辞書型 (Euler angles: `{"seq": "zyx", "angles": [90, 0, 0]}`)

### 2. 数値的安定性と二段階の正規化

* **SVD Normalization (Input Check)**:
外部から読み込んだ行列の歪みが大きい場合、特異値分解（SVD）を用いて回転行列を再構成します。これはグラム・シュミット法よりも安定しており、鏡映（det(R) = -1）を防ぎつつ、最も近い回転行列を復元します。
* **`normalize()` メソッド**:
演算の繰り返しによる微小な浮動小数点誤差をリセットする簡易的な再正規化です。現在のポーズから並進と回転を抽出し、回転表現を再構成することでドリフトを防ぎます。

### 3. リー群・リー代数とねじ理論

数学的な「指数座標」と物理的な「ねじパラメータ」の両方に対応しています。

* **Lie Algebra (se(3))**:
* `exp` / `log` 写像による SE(3) ↔ se(3) 変換。
* 6×6随伴表現（Adjoint Matrix）。
* リー代数上の微小変化に関する Left/Right Jacobian。


* **Screw Parameters**:
* ねじ軸 (Screw Axis) 、ピッチ (Pitch) 、軸上の点 (Point) 、移動・回転量 θ を抽出。



### 4. 高度な演算

* **Composition (`*`)**: ポーズ同士の合成 T₁ ∘ T₂。
* **Point Transformation (`*`)**: 3Dベクトルの座標変換。
* **Inverse**: Rᵀ と −Rᵀt を利用した、行列インバースを回避した T⁻¹ の幾何学的逆変換。

---

## 計算量とアルゴリズム

| 操作 | 時間計算量 | アルゴリズム / 備考 |
| --- | --- | --- |
| **表現へのアクセス** | O(1) | 初回のみ計算、以降はキャッシュヒット。 |
| **逆変換 (Inverse)** | O(1) | Rᵀ と −Rᵀt を利用した幾何学的反転。 |
| **合成 (Composition)** | O(1) | 4×4行列積。 |
| **指数写像 (exp)** | O(1) | Rodrigues' formula (微小角でのテイラー展開を含む)。 |
| **正規化 (SVD)** | O(n³) | 3×3特異値分解による直交化。 |

---

## インストール

```bash
uv add git+https://github.com/kshingent/pose6dof.git

```

---

## 使用例

```python
from pose6dof import Pose6D
import numpy as np

# 1. 様々な入力から初期化（自動判定）
p = Pose6D(np.eye(4))
p_se3 = Pose6D(np.array([0, 0, 0, 0.1, 0, 0])) # 微小なx軸回転

# 2. 遅延評価とキャッシュ
# 以下のプロパティはアクセスした瞬間に計算され、以降は O(1)
quat = p.quat
adj = p.adjoint
screw_pitch = p.screw_pitch

# 3. 演算
p_combined = p * p_se3
v_transformed = p_combined * np.array([1, 0, 0])

# 4. 数値的なドリフトをリセット
p_combined.normalize()

```

---

## ライセンス

MIT License

