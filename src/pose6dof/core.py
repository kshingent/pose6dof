"""
pose6dof - SE(3) Rigid Body Transformation Library

A Python library for intuitive and efficient computation of 3D rigid body transformations.
Built on SciPy's Rotation class, it integrates practical implementations of Lie Group Theory
and Screw Theory for robotics applications.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Union, Tuple
import warnings


class Pose6DOF:
    """
    SE(3) Rigid Body Transformation Class
    
    This class represents 6DOF poses (3D rigid body transformations) using a hybrid
    lazy/eager evaluation system for optimal performance.
    
    Internal representation: 4×4 homogeneous transformation matrix
    ┌         ┐
    │ R  |  t │  (3×3 rotation matrix R, 3×1 translation vector t)
    │----+----│
    │ 0  |  1 │  (bottom row is [0, 0, 0, 1])
    └         ┘
    
    Features:
    - Auto-detection of input formats
    - Lazy evaluation with caching
    - Batch processing support
    - Lie algebra and screw theory
    - Numerical stability with SVD normalization
    """
    
    def __init__(self, *args, pos=None, rot=None, dual_quat=None, eager=False, **kwargs):
        """
        Initialize Pose6DOF with automatic input format detection.
        
        Supported input formats:
        - No arguments: Identity matrix
        - 4×4 matrix: Homogeneous transformation matrix
        - 6D vector: Exponential coordinates (se(3) Lie algebra)
        - pos, rot: Separate translation and rotation
        - dual_quat: Dual quaternion (8D)
        - Pose6DOF object: Copy constructor
        - (N, 4, 4) array: Batch mode
        
        Args:
            *args: Positional arguments for auto-detection
            pos: Translation vector (3D)
            rot: Rotation (3×3 matrix, quaternion, rotation vector, or SciPy Rotation)
            dual_quat: Dual quaternion (8D)
            eager: If True, compute all representations upfront
        """
        # Initialize cache
        self._cache = {}
        self._is_batch = False
        self._batch_size = 0
        
        # Handle different input formats
        if dual_quat is not None:
            # Dual quaternion input
            self._matrix = self._from_dual_quaternion(np.asarray(dual_quat))
        elif pos is not None or rot is not None:
            # Separated position and rotation
            self._matrix = self._from_pos_rot(pos, rot)
        elif len(args) == 0:
            # Default: identity matrix
            self._matrix = np.eye(4)
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Pose6DOF):
                # Copy constructor
                self._matrix = arg._matrix.copy()
                self._is_batch = arg._is_batch
                self._batch_size = arg._batch_size
            else:
                arg = np.asarray(arg)
                if arg.shape == (4, 4):
                    # 4×4 homogeneous transformation matrix
                    self._matrix = self._normalize_matrix(arg)
                elif len(arg.shape) == 3 and arg.shape[1:] == (4, 4):
                    # Batch mode: (N, 4, 4)
                    self._is_batch = True
                    self._batch_size = arg.shape[0]
                    self._matrix = np.array([self._normalize_matrix(m) for m in arg])
                elif arg.shape == (6,):
                    # 6D exponential coordinates (se(3))
                    self._matrix = self._exp_se3(arg)
                else:
                    raise ValueError(f"Unsupported input shape: {arg.shape}")
        elif len(args) == 2:
            # Separated position and rotation
            self._matrix = self._from_pos_rot(args[0], args[1])
        else:
            raise ValueError(f"Unsupported number of arguments: {len(args)}")
        
        # Eager evaluation if requested
        if eager:
            self._compute_all_representations()
    
    @classmethod
    def from_screw_param(cls, axis, point, pitch, theta):
        """
        Create Pose6DOF from screw parameters.
        
        Args:
            axis: Screw axis (3D unit vector)
            point: Point on the axis (3D vector)
            pitch: Screw pitch (translation per radian)
            theta: Rotation/translation amount (radians)
        
        Returns:
            Pose6DOF instance
        """
        axis = np.asarray(axis)
        point = np.asarray(point)
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        # Compute screw motion
        # Rotation part
        rot = R.from_rotvec(axis * theta)
        
        # Translation part: combination of rotation around point and pitch
        # v = (I - R) * (axis × point) + pitch * theta * axis
        rot_matrix = rot.as_matrix()
        moment = np.cross(axis, point)
        translation = (np.eye(3) - rot_matrix) @ moment + pitch * theta * axis
        
        return cls(pos=translation, rot=rot)
    
    @classmethod
    def from_euler(cls, angles, order, degrees=False):
        """
        Create Pose6DOF from Euler angles.
        
        Args:
            angles: Euler angles (3D vector)
            order: Rotation order (e.g., 'xyz', 'zyx')
            degrees: If True, angles are in degrees; otherwise radians
        
        Returns:
            Pose6DOF instance
        """
        rot = R.from_euler(order, angles, degrees=degrees)
        return cls(pos=np.zeros(3), rot=rot)
    
    @classmethod
    def interpolate(cls, pose1, pose2, t):
        """
        Interpolate between two poses using Lie group structure.
        
        This is not simple linear interpolation but preserves the SE(3) group structure
        using exponential map interpolation.
        
        Args:
            pose1: Start pose (Pose6DOF)
            pose2: End pose (Pose6DOF)
            t: Interpolation parameter (0.0 to 1.0)
        
        Returns:
            Interpolated Pose6DOF instance
        """
        # Compute relative transformation
        delta_se3 = pose1.between(pose2)
        
        # Scale the se(3) vector by t
        interpolated_se3 = t * delta_se3
        
        # Apply to pose1
        delta_pose = cls(interpolated_se3)
        return pose1 * delta_pose
    
    def _from_pos_rot(self, pos, rot):
        """Convert separated position and rotation to 4×4 matrix."""
        # Handle position
        if pos is None:
            pos = np.zeros(3)
        else:
            pos = np.asarray(pos)
            if pos.shape != (3,):
                raise ValueError(f"Position must be 3D vector, got shape {pos.shape}")
        
        # Handle rotation
        if rot is None:
            rot_matrix = np.eye(3)
        elif isinstance(rot, R):
            rot_matrix = rot.as_matrix()
        else:
            rot = np.asarray(rot)
            if rot.shape == (3, 3):
                # Rotation matrix
                rot_matrix = self._normalize_rotation_matrix(rot)
            elif rot.shape == (4,):
                # Quaternion (w, x, y, z)
                rot_matrix = R.from_quat(np.roll(rot, -1)).as_matrix()  # Convert to (x, y, z, w)
            elif rot.shape == (3,):
                # Rotation vector / axis-angle
                rot_matrix = R.from_rotvec(rot).as_matrix()
            else:
                raise ValueError(f"Unsupported rotation shape: {rot.shape}")
        
        # Build 4×4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = pos
        return matrix
    
    def _from_dual_quaternion(self, dq):
        """Convert dual quaternion to 4×4 matrix."""
        if dq.shape != (8,):
            raise ValueError(f"Dual quaternion must be 8D, got shape {dq.shape}")
        
        # Real part (rotation quaternion)
        qr = dq[:4]
        # Dual part (translation quaternion)
        qd = dq[4:]
        
        # Normalize real quaternion
        qr = qr / np.linalg.norm(qr)
        
        # Extract rotation
        rot = R.from_quat(np.roll(qr, -1))  # Convert (w, x, y, z) to (x, y, z, w)
        
        # Extract translation: t = 2 * qd * qr*
        qr_conj = np.array([qr[0], -qr[1], -qr[2], -qr[3]])
        t_quat = 2 * self._quat_multiply(qd, qr_conj)
        translation = t_quat[1:]  # Take vector part
        
        return self._from_pos_rot(translation, rot)
    
    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _normalize_matrix(self, matrix):
        """Normalize input matrix using SVD if needed."""
        matrix = np.asarray(matrix, dtype=float)
        
        # Check if it's a valid 4×4 matrix
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4×4, got {matrix.shape}")
        
        # Extract rotation part
        R_input = matrix[:3, :3]
        t = matrix[:3, 3]
        
        # Normalize rotation matrix
        R_normalized = self._normalize_rotation_matrix(R_input)
        
        # Reconstruct matrix
        result = np.eye(4)
        result[:3, :3] = R_normalized
        result[:3, 3] = t
        return result
    
    def _normalize_rotation_matrix(self, R_input):
        """Normalize rotation matrix using SVD."""
        # Check orthogonality
        orthogonality_error = np.linalg.norm(R_input @ R_input.T - np.eye(3))
        
        if orthogonality_error > 1e-6:
            # Use SVD to find nearest rotation matrix
            U, _, Vt = np.linalg.svd(R_input)
            R_normalized = U @ Vt
            
            # Ensure determinant is +1 (not -1, which would be a reflection)
            if np.linalg.det(R_normalized) < 0:
                U[:, -1] *= -1
                R_normalized = U @ Vt
            
            warnings.warn(f"Input rotation matrix was not orthogonal (error: {orthogonality_error:.2e}), "
                         "normalized using SVD.", UserWarning)
        else:
            R_normalized = R_input
        
        return R_normalized
    
    def _exp_se3(self, se3):
        """
        Exponential map: se(3) -> SE(3)
        
        Convert 6D exponential coordinates to 4×4 transformation matrix.
        Uses Rodrigues' formula with Taylor expansion for small angles.
        """
        se3 = np.asarray(se3)
        if se3.shape != (6,):
            raise ValueError(f"se(3) vector must be 6D, got shape {se3.shape}")
        
        # Split into translation and rotation parts
        v = se3[:3]  # Linear velocity
        omega = se3[3:]  # Angular velocity
        
        theta = np.linalg.norm(omega)
        
        if theta < 1e-8:
            # Small angle approximation (Taylor expansion)
            R_exp = np.eye(3)
            V = np.eye(3)
        else:
            # Rodrigues' formula
            omega_normalized = omega / theta
            omega_hat = self._skew_symmetric(omega_normalized)
            
            R_exp = (np.eye(3) + 
                    np.sin(theta) * omega_hat + 
                    (1 - np.cos(theta)) * (omega_hat @ omega_hat))
            
            # Left Jacobian
            V = (np.eye(3) + 
                 (1 - np.cos(theta)) / theta * omega_hat + 
                 (theta - np.sin(theta)) / theta * (omega_hat @ omega_hat))
        
        # Translation part
        t = V @ v
        
        # Construct 4×4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = R_exp
        matrix[:3, 3] = t
        return matrix
    
    def _log_se3(self, matrix):
        """
        Logarithm map: SE(3) -> se(3)
        
        Convert 4×4 transformation matrix to 6D exponential coordinates.
        """
        R_mat = matrix[:3, :3]
        t = matrix[:3, 3]
        
        # Rotation part
        rot = R.from_matrix(R_mat)
        rotvec = rot.as_rotvec()
        theta = np.linalg.norm(rotvec)
        
        if theta < 1e-8:
            # Small angle approximation
            omega = rotvec
            v = t
        else:
            omega = rotvec
            omega_normalized = omega / theta
            omega_hat = self._skew_symmetric(omega_normalized)
            
            # Inverse of left Jacobian
            V_inv = (np.eye(3) - 
                     0.5 * omega_hat + 
                     (1 - theta * np.cos(theta/2) / (2 * np.sin(theta/2))) / theta * 
                     (omega_hat @ omega_hat))
            
            v = V_inv @ t
        
        return np.concatenate([v, omega])
    
    def _skew_symmetric(self, v):
        """Create skew-symmetric matrix from 3D vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def _compute_all_representations(self):
        """Eagerly compute all representations (for eager mode)."""
        _ = self.pos
        _ = self.rot
        _ = self.quat
        _ = self.rotvec
        _ = self.se3
        _ = self.adjoint
        _ = self.screw_axis
        _ = self.screw_pitch
        _ = self.screw_point
        _ = self.dual_quat
    
    # Properties with lazy evaluation and caching
    
    @property
    def matrix(self):
        """Get 4×4 homogeneous transformation matrix."""
        return self._matrix
    
    @property
    def pos(self):
        """Get translation vector (3D)."""
        if 'pos' not in self._cache:
            if self._is_batch:
                self._cache['pos'] = self._matrix[:, :3, 3]
            else:
                self._cache['pos'] = self._matrix[:3, 3]
        return self._cache['pos']
    
    @property
    def rot(self):
        """Get rotation as SciPy Rotation object."""
        if 'rot' not in self._cache:
            if self._is_batch:
                self._cache['rot'] = R.from_matrix(self._matrix[:, :3, :3])
            else:
                self._cache['rot'] = R.from_matrix(self._matrix[:3, :3])
        return self._cache['rot']
    
    @property
    def quat(self):
        """Get rotation as quaternion (w, x, y, z)."""
        if 'quat' not in self._cache:
            quat_xyzw = self.rot.as_quat()
            if self._is_batch:
                # Convert from (x, y, z, w) to (w, x, y, z)
                self._cache['quat'] = np.roll(quat_xyzw, 1, axis=1)
            else:
                # Convert from (x, y, z, w) to (w, x, y, z)
                self._cache['quat'] = np.roll(quat_xyzw, 1)
        return self._cache['quat']
    
    @property
    def rotvec(self):
        """Get rotation as rotation vector (axis-angle)."""
        if 'rotvec' not in self._cache:
            self._cache['rotvec'] = self.rot.as_rotvec()
        return self._cache['rotvec']
    
    @property
    def se3(self):
        """Get pose as 6D exponential coordinates (se(3) Lie algebra)."""
        if 'se3' not in self._cache:
            if self._is_batch:
                self._cache['se3'] = np.array([self._log_se3(m) for m in self._matrix])
            else:
                self._cache['se3'] = self._log_se3(self._matrix)
        return self._cache['se3']
    
    @property
    def adjoint(self):
        """
        Get 6×6 adjoint matrix.
        
        The adjoint representation is used for transforming se(3) vectors.
        """
        if 'adjoint' not in self._cache:
            if self._is_batch:
                n = self._batch_size
                adj = np.zeros((n, 6, 6))
                for i in range(n):
                    R_mat = self._matrix[i, :3, :3]
                    t = self._matrix[i, :3, 3]
                    t_hat = self._skew_symmetric(t)
                    adj[i, :3, :3] = R_mat
                    adj[i, :3, 3:] = t_hat @ R_mat
                    adj[i, 3:, 3:] = R_mat
                self._cache['adjoint'] = adj
            else:
                R_mat = self._matrix[:3, :3]
                t = self._matrix[:3, 3]
                t_hat = self._skew_symmetric(t)
                adj = np.zeros((6, 6))
                adj[:3, :3] = R_mat
                adj[:3, 3:] = t_hat @ R_mat
                adj[3:, 3:] = R_mat
                self._cache['adjoint'] = adj
        return self._cache['adjoint']
    
    @property
    def screw_axis(self):
        """Get screw axis (unit vector)."""
        if 'screw_axis' not in self._cache:
            omega = self.rotvec
            if self._is_batch:
                theta = np.linalg.norm(omega, axis=1, keepdims=True)
                # Avoid division by zero
                theta = np.where(theta < 1e-8, 1.0, theta)
                self._cache['screw_axis'] = omega / theta
            else:
                theta = np.linalg.norm(omega)
                if theta < 1e-8:
                    # Pure translation case
                    v = self.pos
                    v_norm = np.linalg.norm(v)
                    self._cache['screw_axis'] = v / v_norm if v_norm > 1e-8 else np.array([0, 0, 1])
                else:
                    self._cache['screw_axis'] = omega / theta
        return self._cache['screw_axis']
    
    @property
    def screw_pitch(self):
        """Get screw pitch (translation per radian)."""
        if 'screw_pitch' not in self._cache:
            omega = self.rotvec
            v = self.pos
            if self._is_batch:
                theta = np.linalg.norm(omega, axis=1)
                # For pure translation, pitch is infinite
                pitch = np.where(theta < 1e-8, np.inf, 
                               np.einsum('ij,ij->i', v, omega) / (theta ** 2))
                self._cache['screw_pitch'] = pitch
            else:
                theta = np.linalg.norm(omega)
                if theta < 1e-8:
                    # Pure translation: infinite pitch
                    self._cache['screw_pitch'] = np.inf
                else:
                    self._cache['screw_pitch'] = np.dot(v, omega) / (theta ** 2)
        return self._cache['screw_pitch']
    
    @property
    def screw_point(self):
        """Get a point on the screw axis."""
        if 'screw_point' not in self._cache:
            omega = self.rotvec
            v = self.pos
            
            if self._is_batch:
                theta = np.linalg.norm(omega, axis=1, keepdims=True)
                # For pure translation, return origin
                point = np.where(theta[:, 0:1] < 1e-8, 
                               np.zeros_like(v),
                               np.cross(omega, v) / (theta ** 2))
                self._cache['screw_point'] = point
            else:
                theta = np.linalg.norm(omega)
                if theta < 1e-8:
                    # Pure translation: any point on the axis
                    self._cache['screw_point'] = np.zeros(3)
                else:
                    self._cache['screw_point'] = np.cross(omega, v) / (theta ** 2)
        return self._cache['screw_point']
    
    @property
    def dual_quat(self):
        """Get dual quaternion representation (8D)."""
        if 'dual_quat' not in self._cache:
            qr = self.quat  # Real part (rotation quaternion)
            t = self.pos    # Translation
            
            if self._is_batch:
                # Dual part: qd = 0.5 * t_quat * qr
                t_quat = np.zeros((self._batch_size, 4))
                t_quat[:, 1:] = t
                
                # Quaternion multiplication
                qd = np.zeros((self._batch_size, 4))
                for i in range(self._batch_size):
                    qd[i] = 0.5 * self._quat_multiply(t_quat[i], qr[i])
                
                self._cache['dual_quat'] = np.hstack([qr, qd])
            else:
                # Dual part: qd = 0.5 * t_quat * qr
                t_quat = np.array([0, t[0], t[1], t[2]])
                qd = 0.5 * self._quat_multiply(t_quat, qr)
                
                self._cache['dual_quat'] = np.concatenate([qr, qd])
        return self._cache['dual_quat']
    
    # Methods
    
    def normalize(self):
        """
        Normalize the pose to correct numerical drift.
        
        Uses Gram-Schmidt orthogonalization for efficiency (O(n²)).
        This is lighter than SVD and suitable for correcting small numerical errors.
        """
        # Extract current rotation and translation
        if self._is_batch:
            for i in range(self._batch_size):
                R_mat = self._matrix[i, :3, :3]
                
                # Gram-Schmidt orthogonalization
                r1 = R_mat[:, 0]
                r1 = r1 / np.linalg.norm(r1)
                
                r2 = R_mat[:, 1]
                r2 = r2 - np.dot(r2, r1) * r1
                r2 = r2 / np.linalg.norm(r2)
                
                r3 = np.cross(r1, r2)
                
                self._matrix[i, :3, :3] = np.column_stack([r1, r2, r3])
        else:
            R_mat = self._matrix[:3, :3]
            
            # Gram-Schmidt orthogonalization
            r1 = R_mat[:, 0]
            r1 = r1 / np.linalg.norm(r1)
            
            r2 = R_mat[:, 1]
            r2 = r2 - np.dot(r2, r1) * r1
            r2 = r2 / np.linalg.norm(r2)
            
            r3 = np.cross(r1, r2)
            
            self._matrix[:3, :3] = np.column_stack([r1, r2, r3])
        
        # Clear cache since matrix has changed
        self._cache.clear()
        
        return self
    
    def inverse(self):
        """
        Compute inverse transformation using geometric inversion.
        
        Uses R^T and -R^T * t for efficiency, avoiding matrix inverse.
        Time complexity: O(1)
        """
        if self._is_batch:
            inv_matrices = np.zeros_like(self._matrix)
            for i in range(self._batch_size):
                R_mat = self._matrix[i, :3, :3]
                t = self._matrix[i, :3, 3]
                
                R_inv = R_mat.T
                t_inv = -R_inv @ t
                
                inv_matrices[i] = np.eye(4)
                inv_matrices[i, :3, :3] = R_inv
                inv_matrices[i, :3, 3] = t_inv
            
            result = Pose6DOF.__new__(Pose6DOF)
            result._matrix = inv_matrices
            result._cache = {}
            result._is_batch = True
            result._batch_size = self._batch_size
            return result
        else:
            R_mat = self._matrix[:3, :3]
            t = self._matrix[:3, 3]
            
            R_inv = R_mat.T
            t_inv = -R_inv @ t
            
            inv_matrix = np.eye(4)
            inv_matrix[:3, :3] = R_inv
            inv_matrix[:3, 3] = t_inv
            
            return Pose6DOF(inv_matrix)
    
    def between(self, other):
        """
        Compute relative pose between this and another pose.
        
        Returns the se(3) Lie algebra vector: ΔT = self^(-1) * other
        
        Args:
            other: Another Pose6DOF instance
        
        Returns:
            6D se(3) vector representing the relative transformation
        """
        if not isinstance(other, Pose6DOF):
            raise TypeError("other must be a Pose6DOF instance")
        
        # Compute relative transformation
        relative = self.inverse() * other
        return relative.se3
    
    # Operators
    
    def __mul__(self, other):
        """
        Multiplication operator for pose composition and point transformation.
        
        Supports:
        - Pose * Pose: Composition of transformations
        - Pose * Vector: Transform 3D point
        - Pose * Vectors: Transform multiple 3D points
        """
        if isinstance(other, Pose6DOF):
            # Pose composition
            if self._is_batch or other._is_batch:
                # Batch composition
                result_matrix = self._matrix @ other._matrix
            else:
                result_matrix = self._matrix @ other._matrix
            
            result = Pose6DOF.__new__(Pose6DOF)
            result._matrix = result_matrix
            result._cache = {}
            result._is_batch = self._is_batch or other._is_batch
            result._batch_size = max(self._batch_size, other._batch_size)
            return result
        else:
            # Point transformation
            other = np.asarray(other)
            
            if other.shape == (3,):
                # Single 3D point
                point_h = np.append(other, 1)
                if self._is_batch:
                    result = self._matrix @ point_h
                    return result[:, :3]
                else:
                    result = self._matrix @ point_h
                    return result[:3]
            elif len(other.shape) == 2 and other.shape[1] == 3:
                # Multiple 3D points (N, 3)
                points_h = np.hstack([other, np.ones((other.shape[0], 1))])
                if self._is_batch:
                    # Batch pose × multiple points
                    result = np.einsum('bij,nj->bni', self._matrix, points_h)
                    return result[:, :, :3]
                else:
                    result = (self._matrix @ points_h.T).T
                    return result[:, :3]
            else:
                raise ValueError(f"Unsupported shape for point transformation: {other.shape}")
    
    def __repr__(self):
        """String representation."""
        if self._is_batch:
            return f"Pose6DOF(batch_size={self._batch_size})"
        else:
            pos = self.pos
            rotvec = self.rotvec
            return f"Pose6DOF(pos={pos}, rotvec={rotvec})"
    
    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, Pose6DOF):
            return False
        return np.allclose(self._matrix, other._matrix)
