# tests/test_core.py
from pose6dof import Pose6DOF
import numpy as np

def test_init():
    pose = Pose6DOF()
    assert isinstance(pose, Pose6DOF)
    # デフォルトで行列がアイデンティティになるかテスト
    np.testing.assert_array_equal(pose.matrix, np.eye(4))