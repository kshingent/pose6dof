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
| **デフォルト（単位行列）** | 引数なし | 単位行列（identity matrix）を生成。 |
| **Affine Matrix** | 4×4行列 | 同次変換行列として読み込み。回転行列部分の直交性をチェック。 |
| **Exponential Coordinates** | 6次元ベクトル | se(3)リー代数として処理。指数写像を適用。 |
| **Separated (pos, rot)** | 引数が2つ、または名前付き引数 | 並進と回転を個別に処理。 |
| **Pose6DOF Object** | Pose6DOFインスタンス | コピーコンストラクタとして動作。 |
| **Dual Quaternions** | `dual_quat=`キーワード引数 | デュアルクォータニオン（8次元）から変換。 |
| **Batch Matrices** | (N, 4, 4)配列 | N個のポーズを一括処理（バッチモード）。 |

**回転（rot）のサポート形式:**

* 3×3行列 (Rotation Matrix)
* 長さ 4 の配列 (Quaternion: w, x, y, z)
* 長さ 3 の配列 (Rotation Vector / Axis-angle)
* SciPy `Rotation` オブジェクト

**特殊な入力メソッド（自動推定しない）:**

* `from_screw_param(axis, point, pitch, theta)`: ねじパラメータから生成
* `from_euler(angle, order)`: オイラー角から生成（角度単位：ラジアン）

### 2. 内部表現 (Internal Representation)

**Pose6DOFクラスが持つ内部表現は4×4行列`self._matrix`です:**

```
┌         ┐
│ R  |  t │  (3×3回転行列 R, 3×1並進ベクトル t)
│----+----│
│ 0  |  1 │  (最下行は [0, 0, 0, 1])
└         ┘
```

この内部表現により、SE(3)群上の剛体変換を効率的に表現できます。

### 3. 数値的安定性と二段階の正規化

* **SVD Normalization (Input Check)**:
外部から読み込んだ行列の歪みが大きい場合、特異値分解（SVD）を用いて回転行列を再構成します。鏡映（det(R) = -1）を防ぎつつ、最も近い回転行列を復元します。
* **`normalize()` メソッド**:
演算の繰り返しによる微小な浮動小数点誤差をリセットする簡易的な再正規化です。現在のポーズから並進と回転を抽出し、回転表現を再構成することでドリフトを防ぎます。数値誤差の補正を目的としているため、計算コストが軽いグラム・シュミット法を使用します。
  * **変換経路**: `_matrix` → `scipy.Rotation` → `_matrix`
  * **計算コスト**: O(n²)（グラム・シュミット法による直交化）

### 3. リー群・リー代数とねじ理論

数学的な「指数座標」と物理的な「ねじパラメータ」の両方に対応しています。

* **Lie Algebra (se(3))**:
* `exp` / `log` 写像による SE(3) ↔ se(3) 変換。
* 6×6随伴表現（Adjoint Matrix）。
* リー代数上の微小変化に関する Left/Right Jacobian。


* **Screw Parameters**:
* ねじ軸 (Screw Axis) 、ピッチ (Pitch) 、軸上の点 (Point) 、移動・回転量 θ を抽出。
* `from_screw_param()`メソッドで明示的に生成（自動推定しない）。



### 4. 高度な演算

* **Composition (`*`)**: ポーズ同士の合成 T₁ ∘ T₂。
* **Point Transformation (`*`)**: 3Dベクトルの座標変換。
* **Inverse**: Rᵀ と −Rᵀt を利用した、行列インバースを回避した T⁻¹ の幾何学的逆変換。
* **Relative Pose / Between**: 2つのポーズ T₁, T₂ から、相対ポーズ ΔT = T₁⁻¹T₂ をリー代数（6次元ベクトル）として直接取り出す。
* **Interpolation (補間)**: SE(3)上での線形補間。単純な線形補間ではなく、リー群の構造を守った補間（指数写像を利用した補間）を実装。

### 5. バッチ処理 (Batch Operations)

単一のポーズだけでなく、`(N, 4, 4)` のようにNumPyのブロードキャストを利用して一括変換できる機能を持ちます:

```python
# 10個のポーズを一度に処理
matrices = np.random.randn(10, 4, 4)
poses_batch = Pose6DOF(matrices)
```

バッチ処理では、以下の演算もサポートされます:

* **ポーズ同士の合成**: バッチ内の各ポーズに対して別のポーズを合成
* **ベクトル変換**: バッチ内の各ポーズで複数の3Dベクトルを一括変換
* **逆変換**: バッチ内の全ポーズの逆変換を一括計算

---

## 計算量とアルゴリズム

| 操作 | 時間計算量 | アルゴリズム / 備考 |
| --- | --- | --- |
| **表現へのアクセス** | O(1) | 初回のみ計算、以降はキャッシュヒット。 |
| **逆変換 (Inverse)** | O(1) | Rᵀ と −Rᵀt を利用した幾何学的反転。 |
| **合成 (Composition)** | O(1) | 4×4行列積。 |
| **指数写像 (exp)** | O(1) | Rodrigues' formula (微小角でのテイラー展開を含む)。 |
| **対数写像 (log)** | O(1) | SE(3) → se(3) 逆変換。 |
| **正規化 (normalize)** | O(n²) | グラム・シュミット法による直交化。_matrix→R→_matrix経路。 |
| **相対ポーズ (between)** | O(1) | T₁⁻¹T₂ の計算とlog写像。 |
| **補間 (interpolate)** | O(1) | 指数写像を用いたリー群上の補間。 |

---

## インストール

```bash
uv add git+https://github.com/kshingent/pose6dof.git

```

---

## 使用例

```python
from pose6dof import Pose6DOF
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. 様々な入力から初期化（自動判定）

# デフォルト（単位行列）
p = Pose6DOF()

# 4×4同次変換行列から
p_matrix = Pose6DOF(np.eye(4))

# 6次元指数座標（se(3)リー代数）から
p_se3 = Pose6DOF(np.array([0, 0, 0, 0.1, 0, 0]))  # 微小なx軸回転

# 並進と回転を個別に指定
p_pos_rot = Pose6DOF(pos=[1, 2, 3], rot=R.from_euler('xyz', [90, 0, 0], degrees=True))

# デュアルクォータニオンから
p_dual_quat = Pose6DOF(dual_quat=np.array([1, 0, 0, 0, 0, 0.5, 0, 0]))

# Pose6DOFオブジェクトからコピー
p_copy = Pose6DOF(p)

# 2. 特殊な入力メソッド（自動推定しない）

# ねじパラメータから（from_screw_paramメソッドを経由）
p_screw = Pose6DOF.from_screw_param(
    axis=np.array([0, 0, 1]),    # ねじ軸
    point=np.array([0, 0, 0]),   # 軸上の点
    pitch=0.1,                    # ピッチ
    theta=np.pi/4                 # 回転角
)

# オイラー角から（from_eulerメソッドを経由）
p_euler = Pose6DOF.from_euler([np.pi/2, 0, 0], 'xyz')  # 角度はラジアン

# 3. 内部表現
# Pose6DOFクラスが持つ内部表現は4×4行列self._matrixです

# 4. 遅延評価とキャッシュ
# 以下のプロパティはアクセスした瞬間に計算され、以降は O(1)
quat = p.quat
adj = p.adjoint
screw_pitch = p.screw_pitch
dual_quat = p.dual_quat

# 5. 演算
p_combined = p * p_se3
v_transformed = p_combined * np.array([1, 0, 0])

# 6. 数値的なドリフトをリセット
# normalizeメソッドは_matrix→R→_matrixとscipyのRを経由します
# 計算コスト: O(n²)（グラム・シュミット法による直交化）
p_combined.normalize()

# 7. バッチ処理 (Batch Operations)
# (N, 4, 4)の形状でNumPyのブロードキャストを利用した一括変換
poses_batch = Pose6DOF(np.random.randn(10, 4, 4))  # 10個のポーズを一度に処理

# 8. 相対ポーズ (Relative Pose / Between)
# 2つのポーズ T₁, T₂ から、相対ポーズ ΔT = T₁⁻¹T₂ をリー代数（6次元ベクトル）として取り出す
T1 = Pose6DOF(pos=[1, 0, 0])
T2 = Pose6DOF(pos=[2, 0, 0])
delta_se3 = T1.between(T2)  # 6次元ベクトル [v, ω]

# 9. 補間 (Interpolation)
# SE(3)上での線形補間（単純な線形補間ではなく、リー群の構造を守った補間）
# 指数写像を利用した補間の実装
pose_start = Pose6DOF(pos=[0, 0, 0])
pose_end = Pose6DOF(pos=[1, 0, 0])
pose_mid = Pose6DOF.interpolate(pose_start, pose_end, t=0.5)

```

---

## ライセンス

MIT License

