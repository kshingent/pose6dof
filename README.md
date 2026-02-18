# pose6dof

`pose6dof` は、（3次元剛体変換）の計算を「直感的」かつ「高効率」に行うための Python ライブラリです。SciPy の `Rotation` を基盤としつつ、ロボティクスにおけるリー群論（Lie Group Theory）とねじ理論（Screw Theory）の実践的な実装を統合しています。

`Pose6DOF` クラスが持つ内部表現は 4×4 行列 `self._matrix` です。すべての入力形式は最終的にこの同次変換行列に変換され、他の表現形式は遅延評価とキャッシュによって提供されます。

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
| **Dual Quaternions** | 8次元ベクトル | デュアルクォータニオンとして読み込み。実部と双対部から回転と並進を抽出。 |
| **Screw Parameters** | `from_screw_param()` メソッド経由 | ねじ軸、ピッチ、軸上の点、移動量から同次変換行列を構成。自動推定は行わない。 |

**回転（rot）のサポート形式:**

* 3×3行列 (Rotation Matrix)
* 長さ 4 の配列 (Quaternion: w, x, y, z)
* 長さ 3 の配列 (Rotation Vector / Axis-angle)
* SciPy `Rotation` オブジェクト
* 辞書型 (Euler angles: `{"seq": "zyx", "angles": [90, 0, 0]}`)
* `from_euler(angles, seq)` メソッド経由：オイラー角と回転軸順序を指定して回転を生成

### 2. 数値的安定性と二段階の正規化

* **SVD Normalization (Input Check)**:
外部から読み込んだ行列の歪みが大きい場合、特異値分解（SVD）を用いて回転行列を再構成します。これはグラム・シュミット法よりも安定しており、鏡映（det(R) = -1）を防ぎつつ、最も近い回転行列を復元します。
* **`normalize()` メソッド**:
演算の繰り返しによる微小な浮動小数点誤差をリセットする簡易的な再正規化です。現在のポーズから並進と回転を抽出し、回転表現を再構成することでドリフトを防ぎます。
  * **実装の詳細**: `self._matrix` → SciPy の `Rotation` オブジェクト (R) → `self._matrix` という経路で正規化を行います。
  * **計算コスト**: O(1)。回転行列から四元数への変換および四元数の正規化のみを行うため、SVD（O(n³)）よりも高速です。

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

### 5. Batch Operations (バッチ処理)

単一のポーズだけでなく、(N, 4, 4) のような形状で、NumPy のブロードキャストを利用した一括変換が可能です。これにより、複数のポーズを効率的に処理できます。

* 複数の同次変換行列を一度に入力
* ベクトル化された演算による高速処理
* メモリ効率の良いバッチ処理

### 6. Relative Pose / Between (相対ポーズ)

2つのポーズ T₁, T₂ から、相対ポーズ ΔT = T₁⁻¹T₂ を計算し、それをリー代数（6次元ベクトル）として直接取り出す機能を提供します。

* `relative_pose()` メソッドによる相対変換の計算（log 写像により se(3) の6次元ベクトルとして取得）
* se(3) リー代数としての出力により、小さな変化を効率的に表現
* ロボットの制御やビジュアルオドメトリーに有用

### 7. Interpolation (補間)

SE(3) 上での線形補間を実装しています。単純な線形補間ではなく、リー群の構造を守った補間（指数写像を利用した補間）を行います。

* リー代数での線形補間 → 指数写像による SE(3) への変換
* 測地線（geodesic）に沿った補間により、回転と並進の滑らかな遷移を保証
* `interpolate(other, t)` メソッドで、2つのポーズ間の任意の中間点を計算

---

## 計算量とアルゴリズム

| 操作 | 時間計算量 | アルゴリズム / 備考 |
| --- | --- | --- |
| **表現へのアクセス** | O(1) | 初回のみ計算、以降はキャッシュヒット。 |
| **逆変換 (Inverse)** | O(1) | Rᵀ と −Rᵀt を利用した幾何学的反転。 |
| **合成 (Composition)** | O(1) | 4×4行列積。 |
| **指数写像 (exp)** | O(1) | Rodrigues' formula (微小角でのテイラー展開を含む)。 |
| **正規化 (normalize)** | O(1) | `_matrix` → SciPy `Rotation` → `_matrix` による軽量な再正規化。四元数の正規化のみ。 |
| **正規化 (SVD)** | O(n³) | 3×3特異値分解による直交化。入力時の大きな歪みに対してのみ使用。 |

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

# 1. 様々な入力から初期化（自動判定）
p = Pose6DOF(np.eye(4))
p_se3 = Pose6DOF(np.array([0, 0, 0, 0.1, 0, 0])) # 微小なx軸回転

# 2. Dual Quaternionsから初期化
# [w, x, y, z, w_d, x_d, y_d, z_d] 形式（実部4要素 + 双対部4要素）
dq = np.array([1, 0, 0, 0, 0, 0.5, 0, 0])
p_dq = Pose6DOF(dq)

# 3. Screw Parametersから初期化
p_screw = Pose6DOF.from_screw_param(
    axis=np.array([0, 0, 1]),  # ねじ軸
    point=np.array([0, 0, 0]),  # 軸上の点
    pitch=0.1,                   # ピッチ
    theta=np.pi/4                # 移動・回転量
)

# 4. Eulerから初期化
p_euler = Pose6DOF.from_euler(
    angles=[90, 0, 45],
    seq='zyx',
    pos=[1, 2, 3]
)

# 5. 遅延評価とキャッシュ
# 以下のプロパティはアクセスした瞬間に計算され、以降は O(1)
quat = p.quat
adj = p.adjoint
screw_pitch = p.screw_pitch

# 6. 演算
p_combined = p * p_se3
v_transformed = p_combined * np.array([1, 0, 0])

# 7. 数値的なドリフトをリセット
p_combined.normalize()

# 8. バッチ処理
poses = np.stack([np.eye(4) for _ in range(10)])  # 10個の単位行列
p_batch = Pose6DOF(poses)

# 9. 相対ポーズの計算
p1 = Pose6DOF(np.eye(4))
p2 = Pose6DOF.from_euler([0, 0, 45], 'zyx', [1, 0, 0])
delta = p1.relative_pose(p2)  # se(3) の6次元ベクトルとして取得 (shape: (6,))
# 内部で log 写像が自動的に適用されます

# 10. 補間
p_interp = p1.interpolate(p2, t=0.5)  # 中間点

```

---

## ライセンス

MIT License

