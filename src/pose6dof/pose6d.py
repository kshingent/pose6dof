"""
SE(3) Pose representation with hybrid evaluation strategy.

This module provides a Pose6D class that represents 6-DOF poses (position and orientation)
in 3D space. It supports multiple representations and evaluation strategies.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from functools import cached_property
from typing import Union, Optional, Tuple


class Pose6D:
    """
    SE(3) pose representation with hybrid lazy/eager evaluation.
    
    This class represents a 6-DOF pose (3D position + 3D orientation) and maintains
    multiple representations efficiently using lazy evaluation by default, with an
    option for eager evaluation.
    
    Attributes:
        _position (np.ndarray): 3D translation vector
        _rotation (Rotation): Rotation as scipy Rotation object
        _eager (bool): Whether to use eager evaluation
        
    Examples:
        >>> # Create from translation and rotation matrix
        >>> pose = Pose6D(translation=[1, 2, 3], rotation=np.eye(3))
        
        >>> # Create from homogeneous matrix
        >>> matrix = np.eye(4)
        >>> matrix[:3, 3] = [1, 2, 3]
        >>> pose = Pose6D(matrix=matrix)
        
        >>> # Compose two poses
        >>> pose3 = pose1 * pose2
        
        >>> # Transform a point
        >>> point_transformed = pose * point
    """
    
    def __init__(
        self,
        translation: Optional[Union[list, np.ndarray]] = None,
        rotation: Optional[Union[np.ndarray, Rotation]] = None,
        quaternion: Optional[Union[list, np.ndarray]] = None,
        euler: Optional[Union[list, np.ndarray]] = None,
        axis_angle: Optional[Union[list, np.ndarray]] = None,
        matrix: Optional[np.ndarray] = None,
        eager: bool = False
    ):
        """
        Initialize a Pose6D object from various representations.
        
        Args:
            translation: 3D translation vector [x, y, z]
            rotation: 3x3 rotation matrix
            quaternion: Quaternion [x, y, z, w] (scalar-last convention)
            euler: Euler angles [roll, pitch, yaw] in radians
            axis_angle: Axis-angle representation (3D vector, magnitude is angle)
            matrix: 4x4 homogeneous transformation matrix
            eager: If True, pre-compute all representations immediately
            
        Note:
            Only one rotation representation should be provided.
            If matrix is provided, translation and rotation are extracted from it.
        """
        self._eager = eager
        
        # Parse input and set primary state
        if matrix is not None:
            # Extract from 4x4 homogeneous matrix
            matrix = np.asarray(matrix)
            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be 4x4")
            self._position = matrix[:3, 3].copy()
            self._rotation = Rotation.from_matrix(matrix[:3, :3])
        else:
            # Set translation
            if translation is None:
                self._position = np.zeros(3)
            else:
                self._position = np.asarray(translation, dtype=float)
                if self._position.shape != (3,):
                    raise ValueError("Translation must be a 3D vector")
            
            # Set rotation from one of the representations
            if rotation is not None:
                if isinstance(rotation, Rotation):
                    self._rotation = rotation
                else:
                    rotation = np.asarray(rotation)
                    if rotation.shape != (3, 3):
                        raise ValueError("Rotation matrix must be 3x3")
                    self._rotation = Rotation.from_matrix(rotation)
            elif quaternion is not None:
                quaternion = np.asarray(quaternion)
                if quaternion.shape != (4,):
                    raise ValueError("Quaternion must be a 4D vector [x, y, z, w]")
                self._rotation = Rotation.from_quat(quaternion)
            elif euler is not None:
                euler = np.asarray(euler)
                if euler.shape != (3,):
                    raise ValueError("Euler angles must be a 3D vector [roll, pitch, yaw]")
                self._rotation = Rotation.from_euler('xyz', euler)
            elif axis_angle is not None:
                axis_angle = np.asarray(axis_angle)
                if axis_angle.shape != (3,):
                    raise ValueError("Axis-angle must be a 3D vector")
                self._rotation = Rotation.from_rotvec(axis_angle)
            else:
                # Default to identity rotation
                self._rotation = Rotation.from_matrix(np.eye(3))
        
        # If eager evaluation is requested, compute all representations now
        if self._eager:
            _ = self.matrix
            _ = self.quat
            _ = self.euler
            _ = self.se3
    
    @property
    def translation(self) -> np.ndarray:
        """Get the translation vector."""
        return self._position.copy()
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix."""
        return self._rotation.as_matrix()
    
    @cached_property
    def matrix(self) -> np.ndarray:
        """
        Get the 4x4 homogeneous transformation matrix.
        
        Returns:
            4x4 numpy array representing the SE(3) transformation
        """
        T = np.eye(4)
        T[:3, :3] = self._rotation.as_matrix()
        T[:3, 3] = self._position
        return T
    
    @cached_property
    def quat(self) -> np.ndarray:
        """
        Get the quaternion representation [x, y, z, w].
        
        Returns:
            4D numpy array with quaternion in scalar-last convention
        """
        return self._rotation.as_quat()
    
    @cached_property
    def euler(self) -> np.ndarray:
        """
        Get the Euler angles [roll, pitch, yaw] in radians.
        
        Returns:
            3D numpy array with Euler angles (xyz convention)
        """
        return self._rotation.as_euler('xyz')
    
    @cached_property
    def se3(self) -> np.ndarray:
        """
        Get the se(3) Lie algebra representation (exponential coordinates).
        
        Returns:
            6D numpy array [rotation_vector (3), translation (3)]
        """
        rotvec = self._rotation.as_rotvec()
        return np.concatenate([rotvec, self._position])
    
    def normalize(self) -> 'Pose6D':
        """
        Normalize the rotation to ensure it remains orthonormal.
        
        This method ensures numerical stability by re-orthonormalizing the rotation matrix.
        Returns a new Pose6D object with normalized rotation.
        
        Returns:
            A new Pose6D object with normalized rotation
        """
        # The Rotation object from scipy is already normalized, but we recreate
        # it to ensure any numerical drift is corrected
        normalized_rotation = Rotation.from_matrix(self._rotation.as_matrix())
        return Pose6D(
            translation=self._position,
            rotation=normalized_rotation,
            eager=self._eager
        )
    
    def _clear_cache(self):
        """Clear all cached properties."""
        for attr in ['matrix', 'quat', 'euler', 'se3']:
            if attr in self.__dict__:
                delattr(self, attr)
    
    def __mul__(self, other: Union['Pose6D', np.ndarray]) -> Union['Pose6D', np.ndarray]:
        """
        Compose two poses or transform a point/points.
        
        Args:
            other: Either another Pose6D (for composition) or a point/points array
            
        Returns:
            If other is Pose6D: A new composed Pose6D
            If other is array: Transformed point(s)
            
        Examples:
            >>> pose3 = pose1 * pose2  # Pose composition
            >>> point_transformed = pose * point  # Point transformation
        """
        if isinstance(other, Pose6D):
            # Pose composition: self * other
            # New rotation: R1 * R2
            new_rotation = self._rotation * other._rotation
            # New translation: R1 * t2 + t1
            new_translation = self._rotation.apply(other._position) + self._position
            return Pose6D(
                translation=new_translation,
                rotation=new_rotation,
                eager=self._eager
            )
        else:
            # Point transformation
            other = np.asarray(other)
            
            # Handle single point (3,) or multiple points (N, 3)
            if other.ndim == 1:
                if other.shape[0] == 3:
                    # Single 3D point
                    return self._rotation.apply(other) + self._position
                else:
                    raise ValueError("Point must be 3D")
            elif other.ndim == 2:
                if other.shape[1] == 3:
                    # Multiple 3D points (N, 3)
                    return self._rotation.apply(other) + self._position
                else:
                    raise ValueError("Points must have shape (N, 3)")
            else:
                raise ValueError("Invalid point(s) shape")
    
    def __repr__(self) -> str:
        """String representation of the pose."""
        return (f"Pose6D(translation={self._position.tolist()}, "
                f"quaternion={self.quat.tolist()}, eager={self._eager})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"Pose6D:\n"
                f"  Translation: {self._position}\n"
                f"  Rotation (quat): {self.quat}\n"
                f"  Euler (xyz): {np.rad2deg(self.euler)} degrees")
    
    def inverse(self) -> 'Pose6D':
        """
        Compute the inverse of this pose.
        
        Returns:
            A new Pose6D representing the inverse transformation
        """
        # R^T for rotation inverse
        inv_rotation = self._rotation.inv()
        # -R^T * t for translation
        inv_translation = -inv_rotation.apply(self._position)
        return Pose6D(
            translation=inv_translation,
            rotation=inv_rotation,
            eager=self._eager
        )
    
    def to_dict(self) -> dict:
        """
        Convert pose to a dictionary with all representations.
        
        Returns:
            Dictionary containing all pose representations
        """
        return {
            'translation': self._position.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist(),
            'quaternion': self.quat.tolist(),
            'euler': self.euler.tolist(),
            'matrix': self.matrix.tolist(),
            'se3': self.se3.tolist()
        }
    
    @classmethod
    def identity(cls, eager: bool = False) -> 'Pose6D':
        """
        Create an identity pose (no translation, no rotation).
        
        Args:
            eager: Whether to use eager evaluation
            
        Returns:
            Identity Pose6D object
        """
        return cls(eager=eager)
    
    @classmethod
    def from_se3(cls, se3_vector: np.ndarray, eager: bool = False) -> 'Pose6D':
        """
        Create a pose from se(3) Lie algebra representation.
        
        Args:
            se3_vector: 6D vector [rotation_vector (3), translation (3)]
            eager: Whether to use eager evaluation
            
        Returns:
            Pose6D object
        """
        se3_vector = np.asarray(se3_vector)
        if se3_vector.shape != (6,):
            raise ValueError("se3 vector must be 6D")
        
        rotvec = se3_vector[:3]
        translation = se3_vector[3:]
        
        return cls(
            translation=translation,
            axis_angle=rotvec,
            eager=eager
        )
