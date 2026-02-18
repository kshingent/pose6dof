"""
Example usage of the Pose6D class.

This script demonstrates the main features of the Pose6D class including:
1. Different initialization methods
2. Lazy vs eager evaluation
3. Pose composition
4. Point transformation
5. Normalization
"""

import numpy as np
import sys
sys.path.insert(0, '/home/runner/work/pose6dof/pose6dof/src')

from pose6dof import Pose6D


def main():
    print("=" * 80)
    print("Pose6D Example Usage")
    print("=" * 80)
    
    # 1. Create poses with different initialization methods
    print("\n1. Different Initialization Methods:")
    print("-" * 80)
    
    # Identity pose
    pose_identity = Pose6D.identity()
    print(f"Identity pose:\n{pose_identity}\n")
    
    # From translation and rotation matrix
    translation = [1, 2, 3]
    rotation = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=float)  # 90° rotation around z-axis
    pose1 = Pose6D(translation=translation, rotation=rotation)
    print(f"Pose from translation and rotation matrix:\n{pose1}\n")
    
    # From quaternion
    quat = [0, 0, 0.7071068, 0.7071068]  # 90° around z-axis
    pose2 = Pose6D(translation=[1, 0, 0], quaternion=quat)
    print(f"Pose from quaternion:\n{pose2}\n")
    
    # From Euler angles
    euler = [0, 0, np.pi/4]  # 45° around z-axis
    pose3 = Pose6D(translation=[0, 1, 0], euler=euler)
    print(f"Pose from Euler angles:\n{pose3}\n")
    
    # From 4x4 homogeneous matrix
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    matrix[:3, :3] = rotation
    pose4 = Pose6D(matrix=matrix)
    print(f"Pose from 4x4 matrix:\n{pose4}\n")
    
    # From se(3) vector
    se3_vec = np.array([0, 0, np.pi/6, 1, 1, 1])
    pose5 = Pose6D.from_se3(se3_vec)
    print(f"Pose from se(3) vector:\n{pose5}\n")
    
    # 2. Lazy vs Eager Evaluation
    print("\n2. Lazy vs Eager Evaluation:")
    print("-" * 80)
    
    pose_lazy = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3], eager=False)
    print(f"Lazy evaluation - cached properties before access:")
    print(f"  'matrix' in __dict__: {'matrix' in pose_lazy.__dict__}")
    print(f"  'quat' in __dict__: {'quat' in pose_lazy.__dict__}")
    
    # Access a property
    _ = pose_lazy.matrix
    print(f"\nAfter accessing 'matrix':")
    print(f"  'matrix' in __dict__: {'matrix' in pose_lazy.__dict__}")
    print(f"  'quat' in __dict__: {'quat' in pose_lazy.__dict__}")
    
    pose_eager = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3], eager=True)
    print(f"\nEager evaluation - all properties pre-computed:")
    print(f"  'matrix' in __dict__: {'matrix' in pose_eager.__dict__}")
    print(f"  'quat' in __dict__: {'quat' in pose_eager.__dict__}")
    print(f"  'euler' in __dict__: {'euler' in pose_eager.__dict__}")
    print(f"  'se3' in __dict__: {'se3' in pose_eager.__dict__}")
    
    # 3. Pose Composition
    print("\n\n3. Pose Composition:")
    print("-" * 80)
    
    pose_a = Pose6D(translation=[1, 0, 0])
    pose_b = Pose6D(translation=[0, 1, 0], euler=[0, 0, np.pi/2])
    
    print(f"Pose A translation: {pose_a.translation}")
    print(f"Pose B translation: {pose_b.translation}")
    print(f"Pose B rotation (euler): {np.rad2deg(pose_b.euler)} degrees")
    
    pose_c = pose_a * pose_b
    print(f"\nComposed Pose C = A * B:")
    print(f"  Translation: {pose_c.translation}")
    print(f"  Rotation (euler): {np.rad2deg(pose_c.euler)} degrees")
    
    # 4. Point Transformation
    print("\n\n4. Point Transformation:")
    print("-" * 80)
    
    pose = Pose6D(translation=[2, 3, 4], euler=[0, 0, np.pi/2])
    
    # Transform a single point
    point = np.array([1, 0, 0])
    transformed_point = pose * point
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")
    
    # Transform multiple points
    points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    transformed_points = pose * points
    print(f"\nOriginal points:\n{points}")
    print(f"Transformed points:\n{transformed_points}")
    
    # 5. Inverse Transform
    print("\n\n5. Inverse Transform:")
    print("-" * 80)
    
    pose = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
    pose_inv = pose.inverse()
    
    print(f"Original pose translation: {pose.translation}")
    print(f"Inverse pose translation: {pose_inv.translation}")
    
    # Verify: pose * pose_inv = identity
    identity = pose * pose_inv
    print(f"\nPose * Inverse = Identity:")
    print(f"  Translation: {identity.translation}")
    print(f"  Rotation matrix close to I: {np.allclose(identity.rotation_matrix, np.eye(3))}")
    
    # 6. Normalization
    print("\n\n6. Normalization for Numerical Stability:")
    print("-" * 80)
    
    pose = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
    normalized_pose = pose.normalize()
    
    R = normalized_pose.rotation_matrix
    print(f"Rotation matrix is orthonormal:")
    print(f"  R^T * R close to I: {np.allclose(R.T @ R, np.eye(3))}")
    print(f"  det(R) = {np.linalg.det(R):.10f} (should be 1.0)")
    
    # 7. Different Representations
    print("\n\n7. Access Different Representations:")
    print("-" * 80)
    
    pose = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
    
    print(f"Translation: {pose.translation}")
    print(f"Rotation matrix shape: {pose.rotation_matrix.shape}")
    print(f"Quaternion [x,y,z,w]: {pose.quat}")
    print(f"Euler angles [r,p,y] (rad): {pose.euler}")
    print(f"Euler angles [r,p,y] (deg): {np.rad2deg(pose.euler)}")
    print(f"se(3) vector: {pose.se3}")
    print(f"4x4 matrix shape: {pose.matrix.shape}")
    
    # 8. Conversion to Dictionary
    print("\n\n8. Convert to Dictionary:")
    print("-" * 80)
    
    pose = Pose6D(translation=[1, 2, 3], euler=[0, 0, np.pi/4])
    pose_dict = pose.to_dict()
    print(f"Dictionary keys: {list(pose_dict.keys())}")
    print(f"Translation from dict: {pose_dict['translation']}")
    print(f"Quaternion from dict: {pose_dict['quaternion']}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
