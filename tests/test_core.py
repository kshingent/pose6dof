# tests/test_core.py
from pose6dof import Pose6DOF
import numpy as np
from scipy.spatial.transform import Rotation as R

def test_init():
    """Test default initialization (identity matrix)."""
    pose = Pose6DOF()
    assert isinstance(pose, Pose6DOF)
    # Test that matrix is identity by default
    np.testing.assert_array_equal(pose.matrix, np.eye(4))

def test_init_from_matrix():
    """Test initialization from 4×4 matrix."""
    mat = np.eye(4)
    mat[:3, 3] = [1, 2, 3]
    pose = Pose6DOF(mat)
    np.testing.assert_array_almost_equal(pose.pos, [1, 2, 3])

def test_init_from_se3():
    """Test initialization from 6D exponential coordinates."""
    se3 = np.array([0, 0, 0, 0.1, 0, 0])  # small x-axis rotation
    pose = Pose6DOF(se3)
    assert pose.matrix.shape == (4, 4)

def test_init_from_pos_rot():
    """Test initialization from separated pos and rot."""
    pos = np.array([1, 2, 3])
    rot = R.from_euler('xyz', [90, 0, 0], degrees=True)
    pose = Pose6DOF(pos=pos, rot=rot)
    np.testing.assert_array_almost_equal(pose.pos, pos)

def test_init_from_dual_quat():
    """Test initialization from dual quaternion."""
    # Identity rotation, translation [1, 0, 0]
    dq = np.array([1, 0, 0, 0, 0, 0.5, 0, 0])
    pose = Pose6DOF(dual_quat=dq)
    assert pose.matrix.shape == (4, 4)

def test_init_pose6dof_copy():
    """Test copy constructor."""
    pose1 = Pose6DOF(pos=[1, 2, 3])
    pose2 = Pose6DOF(pose1)
    np.testing.assert_array_equal(pose1.matrix, pose2.matrix)

def test_from_screw_param():
    """Test creation from screw parameters."""
    axis = np.array([0, 0, 1])
    point = np.array([0, 0, 0])
    pitch = 0.1
    theta = np.pi / 4
    pose = Pose6DOF.from_screw_param(axis, point, pitch, theta)
    assert pose.matrix.shape == (4, 4)

def test_from_euler():
    """Test creation from Euler angles."""
    pose = Pose6DOF.from_euler([90, 0, 0], 'xyz', degrees=True, pos=[1, 2, 3])
    assert pose.matrix.shape == (4, 4)
    np.testing.assert_array_almost_equal(pose.pos, [1, 2, 3])

def test_normalize():
    """Test normalize method."""
    pose = Pose6DOF(pos=[1, 2, 3])
    pose.normalize()
    # Should still be identity rotation
    np.testing.assert_array_almost_equal(pose.matrix[:3, :3], np.eye(3))

def test_properties():
    """Test property accessors."""
    pose = Pose6DOF(pos=[1, 2, 3], rot=R.from_euler('z', 90, degrees=True))
    
    # Test pos
    np.testing.assert_array_almost_equal(pose.pos, [1, 2, 3])
    
    # Test quat
    assert pose.quat.shape == (4,)
    
    # Test rotvec
    assert pose.rotvec.shape == (3,)
    
    # Test se3
    assert pose.se3.shape == (6,)
    
    # Test adjoint
    assert pose.adjoint.shape == (6, 6)
    
    # Test dual_quat
    assert pose.dual_quat.shape == (8,)

def test_inverse():
    """Test inverse transformation."""
    pose = Pose6DOF(pos=[1, 2, 3], rot=R.from_euler('z', 90, degrees=True))
    pose_inv = pose.inverse()
    
    # Compose should give identity
    identity = pose * pose_inv
    np.testing.assert_array_almost_equal(identity.matrix, np.eye(4), decimal=10)

def test_composition():
    """Test pose composition."""
    pose1 = Pose6DOF(pos=[1, 0, 0])
    pose2 = Pose6DOF(pos=[0, 1, 0])
    pose_composed = pose1 * pose2
    
    # Result should be translation [1, 1, 0]
    np.testing.assert_array_almost_equal(pose_composed.pos, [1, 1, 0])

def test_point_transformation():
    """Test point transformation."""
    pose = Pose6DOF(pos=[1, 2, 3])
    point = np.array([0, 0, 0])
    transformed = pose * point
    np.testing.assert_array_almost_equal(transformed, [1, 2, 3])

def test_between():
    """Test relative pose computation."""
    pose1 = Pose6DOF(pos=[1, 0, 0])
    pose2 = Pose6DOF(pos=[2, 0, 0])
    delta = pose1.between(pose2)
    
    # Should be a 6D twist
    assert delta.shape == (6,)
    # Translation should be [1, 0, 0] in pose1's frame
    np.testing.assert_array_almost_equal(delta[:3], [1, 0, 0])

def test_interpolate():
    """Test SE(3) interpolation."""
    pose1 = Pose6DOF(pos=[0, 0, 0])
    pose2 = Pose6DOF(pos=[1, 0, 0])
    
    # Interpolate at t=0.5
    pose_mid = Pose6DOF.interpolate(pose1, pose2, 0.5)
    np.testing.assert_array_almost_equal(pose_mid.pos, [0.5, 0, 0])

def test_batch_operations():
    """Test batch operations with (N, 4, 4) arrays."""
    # Create batch of matrices
    matrices = np.stack([np.eye(4) for _ in range(5)])
    matrices[:, :3, 3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
    
    pose_batch = Pose6DOF(matrices)
    assert pose_batch.matrix.shape == (5, 4, 4)

def test_batch_composition_not_implemented():
    """Test that batch pose composition raises NotImplementedError."""
    import pytest
    
    matrices1 = np.stack([np.eye(4) for _ in range(2)])
    matrices2 = np.stack([np.eye(4) for _ in range(2)])
    
    pose_batch1 = Pose6DOF(matrices1)
    pose_batch2 = Pose6DOF(matrices2)
    
    with pytest.raises(NotImplementedError):
        _ = pose_batch1 * pose_batch2