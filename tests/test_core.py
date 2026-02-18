# tests/test_core.py
from pose6dof import Pose6D
import numpy as np

def test_init():
    pose = Pose6D()
    assert isinstance(pose, Pose6D)
    # デフォルトで行列がアイデンティティになるかテスト
    np.testing.assert_array_equal(pose.matrix, np.eye(4))