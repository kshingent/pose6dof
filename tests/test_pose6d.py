"""
Unit tests for the Pose6D class.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose6dof import Pose6D


class TestPose6DInitialization:
    """Tests for various initialization methods."""
    
    def test_default_initialization(self):
        """Test creating an identity pose."""
        pose = Pose6D()
        np.testing.assert_array_almost_equal(pose.translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(pose.rotation_matrix, np.eye(3))
    
    def test_identity_class_method(self):
        """Test identity class method."""
        pose = Pose6D.identity()
        np.testing.assert_array_almost_equal(pose.translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(pose.rotation_matrix, np.eye(3))
    
    def test_translation_only(self):
        """Test initialization with only translation."""
        translation = [1, 2, 3]
        pose = Pose6D(translation=translation)
        np.testing.assert_array_almost_equal(pose.translation, translation)
        np.testing.assert_array_almost_equal(pose.rotation_matrix, np.eye(3))
    
    def test_rotation_matrix(self):
        """Test initialization with rotation matrix."""
        # 90 degree rotation around z-axis
        rot_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
        pose = Pose6D(rotation=rot_matrix)
        np.testing.assert_array_almost_equal(pose.rotation_matrix, rot_matrix)
    
    def test_quaternion_initialization(self):
        """Test initialization with quaternion."""
        # Identity quaternion [x, y, z, w]
        quat = [0, 0, 0, 1]
        pose = Pose6D(quaternion=quat)
        np.testing.assert_array_almost_equal(pose.quat, quat)
    
    def test_euler_initialization(self):
        """Test initialization with Euler angles."""
        euler = [0, 0, np.pi/2]  # 90 degrees around z
        pose = Pose6D(euler=euler)
        np.testing.assert_array_almost_equal(pose.euler, euler)
    
    def test_axis_angle_initialization(self):
        """Test initialization with axis-angle."""
        axis_angle = [0, 0, np.pi/2]  # 90 degrees around z
        pose = Pose6D(axis_angle=axis_angle)
        expected_rot = Rotation.from_rotvec(axis_angle).as_matrix()
        np.testing.assert_array_almost_equal(pose.rotation_matrix, expected_rot)
    
    def test_matrix_initialization(self):
        """Test initialization from 4x4 homogeneous matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        matrix[:3, :3] = Rotation.from_euler('xyz', [0, 0, np.pi/4]).as_matrix()
        
        pose = Pose6D(matrix=matrix)
        np.testing.assert_array_almost_equal(pose.matrix, matrix)
        np.testing.assert_array_almost_equal(pose.translation, [1, 2, 3])
    
    def test_from_se3(self):
        """Test initialization from se(3) vector."""
        se3_vec = np.array([0.1, 0.2, 0.3, 1, 2, 3])
        pose = Pose6D.from_se3(se3_vec)
        np.testing.assert_array_almost_equal(pose.se3, se3_vec)
    
    def test_scipy_rotation_object(self):
        """Test initialization with scipy Rotation object."""
        rot = Rotation.from_euler('xyz', [0, 0, np.pi/2])
        pose = Pose6D(rotation=rot)
        np.testing.assert_array_almost_equal(
            pose.rotation_matrix,
            rot.as_matrix()
        )


class TestPose6DLazyEvaluation:
    """Tests for lazy evaluation using cached_property."""
    
    def test_lazy_evaluation(self):
        """Test that properties are computed lazily."""
        pose = Pose6D(translation=[1, 2, 3], eager=False)
        
        # Before accessing, cached properties shouldn't exist in __dict__
        assert 'matrix' not in pose.__dict__
        assert 'quat' not in pose.__dict__
        assert 'euler' not in pose.__dict__
        assert 'se3' not in pose.__dict__
        
        # Access matrix - should now be cached
        _ = pose.matrix
        assert 'matrix' in pose.__dict__
        
        # Others should still not be computed
        assert 'quat' not in pose.__dict__
    
    def test_eager_evaluation(self):
        """Test that eager mode pre-computes all representations."""
        pose = Pose6D(translation=[1, 2, 3], eager=True)
        
        # All cached properties should be pre-computed
        assert 'matrix' in pose.__dict__
        assert 'quat' in pose.__dict__
        assert 'euler' in pose.__dict__
        assert 'se3' in pose.__dict__


class TestPose6DRepresentations:
    """Tests for different pose representations."""
    
    def test_matrix_property(self):
        """Test the 4x4 matrix representation."""
        translation = [1, 2, 3]
        pose = Pose6D(translation=translation)
        
        matrix = pose.matrix
        assert matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(matrix[:3, 3], translation)
        np.testing.assert_array_almost_equal(matrix[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(matrix[3, :], [0, 0, 0, 1])
    
    def test_quaternion_property(self):
        """Test quaternion representation."""
        euler = [0, 0, np.pi/2]
        pose = Pose6D(euler=euler)
        quat = pose.quat
        
        assert quat.shape == (4,)
        # Verify by converting back
        rot = Rotation.from_quat(quat)
        np.testing.assert_array_almost_equal(
            rot.as_euler('xyz'),
            euler
        )
    
    def test_euler_property(self):
        """Test Euler angles representation."""
        euler_input = [0.1, 0.2, 0.3]
        pose = Pose6D(euler=euler_input)
        np.testing.assert_array_almost_equal(pose.euler, euler_input)
    
    def test_se3_property(self):
        """Test se(3) representation."""
        translation = [1, 2, 3]
        axis_angle = [0.1, 0.2, 0.3]
        pose = Pose6D(translation=translation, axis_angle=axis_angle)
        
        se3 = pose.se3
        assert se3.shape == (6,)
        np.testing.assert_array_almost_equal(se3[:3], axis_angle)
        np.testing.assert_array_almost_equal(se3[3:], translation)


class TestPose6DOperators:
    """Tests for operator overloading."""
    
    def test_pose_composition(self):
        """Test composing two poses using * operator."""
        # Create two poses
        pose1 = Pose6D(translation=[1, 0, 0])
        pose2 = Pose6D(translation=[0, 1, 0])
        
        # Compose them
        pose3 = pose1 * pose2
        
        # Result should have translation [1, 1, 0]
        np.testing.assert_array_almost_equal(
            pose3.translation,
            [1, 1, 0]
        )
    
    def test_pose_composition_with_rotation(self):
        """Test pose composition with rotation."""
        # First pose: translate by [1, 0, 0]
        pose1 = Pose6D(translation=[1, 0, 0])
        
        # Second pose: rotate 90 degrees around z, then translate [1, 0, 0]
        pose2 = Pose6D(
            translation=[1, 0, 0],
            euler=[0, 0, np.pi/2]
        )
        
        # Compose: T3 = pose1 * pose2
        # R3 = R1 * R2 = I * R2 = R2 (90° rotation around z)
        # T3 = R1 * T2 + T1 = I * [1,0,0] + [1,0,0] = [2, 0, 0]
        pose3 = pose1 * pose2
        
        expected_translation = [2, 0, 0]
        np.testing.assert_array_almost_equal(
            pose3.translation,
            expected_translation,
            decimal=5
        )
        
        # Verify rotation is preserved
        expected_rotation = Rotation.from_euler('xyz', [0, 0, np.pi/2]).as_matrix()
        np.testing.assert_array_almost_equal(
            pose3.rotation_matrix,
            expected_rotation,
            decimal=5
        )
    
    def test_point_transformation_single(self):
        """Test transforming a single point."""
        pose = Pose6D(translation=[1, 2, 3])
        point = np.array([1, 0, 0])
        
        transformed = pose * point
        np.testing.assert_array_almost_equal(transformed, [2, 2, 3])
    
    def test_point_transformation_multiple(self):
        """Test transforming multiple points."""
        pose = Pose6D(translation=[1, 2, 3])
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        transformed = pose * points
        expected = np.array([
            [2, 2, 3],
            [1, 3, 3],
            [1, 2, 4]
        ])
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_point_transformation_with_rotation(self):
        """Test point transformation with rotation."""
        # 90 degree rotation around z-axis
        pose = Pose6D(
            translation=[1, 0, 0],
            euler=[0, 0, np.pi/2]
        )
        point = np.array([1, 0, 0])
        
        # Point [1, 0, 0] rotated 90° around z becomes [0, 1, 0]
        # Then translated by [1, 0, 0] becomes [1, 1, 0]
        transformed = pose * point
        np.testing.assert_array_almost_equal(transformed, [1, 1, 0], decimal=5)


class TestPose6DNormalize:
    """Tests for the normalize method."""
    
    def test_normalize_identity(self):
        """Test normalizing an identity pose."""
        pose = Pose6D()
        normalized = pose.normalize()
        
        np.testing.assert_array_almost_equal(
            normalized.rotation_matrix,
            np.eye(3)
        )
    
    def test_normalize_preserves_properties(self):
        """Test that normalize preserves pose properties."""
        translation = [1, 2, 3]
        euler = [0.1, 0.2, 0.3]
        pose = Pose6D(translation=translation, euler=euler)
        
        normalized = pose.normalize()
        
        np.testing.assert_array_almost_equal(
            normalized.translation,
            translation
        )
        np.testing.assert_array_almost_equal(
            normalized.euler,
            euler
        )
    
    def test_normalize_orthonormalizes(self):
        """Test that normalize ensures orthonormality."""
        # Create a slightly non-orthonormal matrix
        rot_matrix = np.array([
            [1.001, 0, 0],
            [0, 0.999, 0],
            [0, 0, 1]
        ])
        
        pose = Pose6D(rotation=rot_matrix)
        normalized = pose.normalize()
        
        # Check orthonormality: R^T * R = I
        R = normalized.rotation_matrix
        np.testing.assert_array_almost_equal(
            R.T @ R,
            np.eye(3),
            decimal=10
        )
        
        # Check determinant is 1
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-10


class TestPose6DInverse:
    """Tests for the inverse method."""
    
    def test_inverse_identity(self):
        """Test inverse of identity is identity."""
        pose = Pose6D.identity()
        inv_pose = pose.inverse()
        
        np.testing.assert_array_almost_equal(
            inv_pose.translation,
            [0, 0, 0]
        )
        np.testing.assert_array_almost_equal(
            inv_pose.rotation_matrix,
            np.eye(3)
        )
    
    def test_inverse_translation(self):
        """Test inverse of pure translation."""
        pose = Pose6D(translation=[1, 2, 3])
        inv_pose = pose.inverse()
        
        np.testing.assert_array_almost_equal(
            inv_pose.translation,
            [-1, -2, -3]
        )
    
    def test_inverse_composition(self):
        """Test that pose * inverse = identity."""
        translation = [1, 2, 3]
        euler = [0.1, 0.2, 0.3]
        pose = Pose6D(translation=translation, euler=euler)
        
        inv_pose = pose.inverse()
        identity = pose * inv_pose
        
        np.testing.assert_array_almost_equal(
            identity.translation,
            [0, 0, 0],
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            identity.rotation_matrix,
            np.eye(3),
            decimal=10
        )


class TestPose6DUtilities:
    """Tests for utility methods."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pose = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
        d = pose.to_dict()
        
        assert 'translation' in d
        assert 'rotation_matrix' in d
        assert 'quaternion' in d
        assert 'euler' in d
        assert 'matrix' in d
        assert 'se3' in d
        
        np.testing.assert_array_almost_equal(d['translation'], [1, 2, 3])
    
    def test_repr(self):
        """Test __repr__ method."""
        pose = Pose6D(translation=[1, 2, 3])
        repr_str = repr(pose)
        assert 'Pose6D' in repr_str
        assert 'translation' in repr_str
    
    def test_str(self):
        """Test __str__ method."""
        pose = Pose6D(translation=[1, 2, 3])
        str_repr = str(pose)
        assert 'Pose6D' in str_repr
        assert 'Translation' in str_repr


class TestPose6DEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_invalid_translation_shape(self):
        """Test that invalid translation shape raises error."""
        with pytest.raises(ValueError, match="3D vector"):
            Pose6D(translation=[1, 2])
    
    def test_invalid_rotation_shape(self):
        """Test that invalid rotation matrix shape raises error."""
        with pytest.raises(ValueError, match="3x3"):
            Pose6D(rotation=np.eye(2))
    
    def test_invalid_quaternion_shape(self):
        """Test that invalid quaternion shape raises error."""
        with pytest.raises(ValueError, match="4D vector"):
            Pose6D(quaternion=[0, 0, 1])
    
    def test_invalid_matrix_shape(self):
        """Test that invalid matrix shape raises error."""
        with pytest.raises(ValueError, match="4x4"):
            Pose6D(matrix=np.eye(3))
    
    def test_invalid_point_shape(self):
        """Test that invalid point shape raises error."""
        pose = Pose6D()
        with pytest.raises(ValueError, match="3D"):
            _ = pose * np.array([1, 2])
    
    def test_invalid_se3_shape(self):
        """Test that invalid se3 vector raises error."""
        with pytest.raises(ValueError, match="6D"):
            Pose6D.from_se3([1, 2, 3])


class TestPose6DCacheConsistency:
    """Tests for cache consistency."""
    
    def test_cache_cleared_on_normalization(self):
        """Test that cache is properly managed in new instances."""
        # This is more of a design test - ensure new objects don't share cache
        pose1 = Pose6D(translation=[1, 2, 3], eager=False)
        _ = pose1.matrix  # Cache matrix
        
        pose2 = pose1.normalize()
        # pose2 should be a new object with its own cache
        assert pose1 is not pose2
        
        # If eager=False was preserved, pose2's cache should be empty initially
        # But since normalize creates a new object, it should have its own cache
    
    def test_representations_consistent(self):
        """Test that all representations are mathematically consistent."""
        translation = [1, 2, 3]
        euler = [0.1, 0.2, 0.3]
        pose = Pose6D(translation=translation, euler=euler)
        
        # Extract matrix and reconstruct
        matrix = pose.matrix
        pose2 = Pose6D(matrix=matrix)
        
        np.testing.assert_array_almost_equal(
            pose.translation,
            pose2.translation
        )
        np.testing.assert_array_almost_equal(
            pose.rotation_matrix,
            pose2.rotation_matrix
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
