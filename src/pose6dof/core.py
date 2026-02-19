"""
pose6dof - SE(3) Rigid Body Transformation Library

A Python library for intuitive and efficient computation of 3D rigid body transformations.
Built on SciPy's Rotation class, it integrates practical implementations of Lie Group Theory
and Screw Theory for robotics applications.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
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
    
    def __init__(self, *args, eager: bool = False) -> None:
        """
        Initialize Pose6DOF with automatic input format detection.
        
        Supported input formats:
        - No arguments: Identity matrix
        - Single positional argument:
          - 4×4 matrix: Homogeneous transformation matrix
          - 6D vector: Exponential coordinates (se(3) Lie algebra)
          - 8D vector: Dual quaternion
          - Pose6DOF object: Copy constructor
          - (N, 4, 4) array: Batch mode
        - Two positional arguments: (position, rotation)
          - position: 3D translation vector
          - rotation: 3×3 matrix, 4D quaternion, 3D rotation vector, or SciPy Rotation
        
        Args:
            *args: Positional arguments for auto-detection
            eager: If True, compute all representations upfront
        """
        # Initialize cache
        self._cache: dict = {}
        self._is_batch: bool = False
        self._batch_size: int = 0
        
        # Handle different input formats based on auto-detection
        if len(args) == 0:
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
                    self._matrix = self._normalize_batch_matrices(arg)
                elif arg.shape == (6,):
                    # 6D exponential coordinates (se(3))
                    self._matrix = self._exp_se3(arg)
                elif arg.shape == (8,):
                    # Dual quaternion input
                    self._matrix = self._from_dual_quaternion(arg)
                else:
                    raise ValueError(f"Unsupported input shape: {arg.shape}")
        elif len(args) == 2:
            # Separated position and rotation (auto-detected from two arguments)
            self._matrix = self._from_pos_rot(args[0], args[1])
        else:
            raise ValueError(f"Unsupported number of arguments: {len(args)}")
        
        # Eager evaluation if requested
        if eager:
            self._compute_all_representations()
    
    @classmethod
    def from_screw_param(cls, axis: np.ndarray | list, point: np.ndarray | list, 
                        pitch: float, theta: float) -> 'Pose6DOF':
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
        
        return cls(translation, rot)
    
    @classmethod
    def from_euler(cls, angles: np.ndarray | list, order: str, 
                   pos: np.ndarray | list, degrees: bool = False) -> 'Pose6DOF':
        """
        Create Pose6DOF from Euler angles.
        
        Args:
            angles: Euler angles (3D vector)
            order: Rotation order (e.g., 'xyz', 'zyx')
            pos: Position vector (3D). Required.
            degrees: If True, angles are in degrees; otherwise radians
        
        Returns:
            Pose6DOF instance
        """
        rot = R.from_euler(order, angles, degrees=degrees)
        pos = np.asarray(pos)
        if pos.shape != (3,):
            raise ValueError(f"Position must be a 3D vector, got shape {pos.shape}")
        return cls(pos, rot)
    
    @classmethod
    def interpolate(cls, pose1: 'Pose6DOF', pose2: 'Pose6DOF', t: float) -> 'Pose6DOF':
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
    
    def _from_pos_rot(self, pos: np.ndarray | list | None, 
                      rot: np.ndarray | list | R | None) -> np.ndarray:
        """
        Convert separated position and rotation to 4×4 matrix.
        
        If pos is None, defaults to zero translation [0, 0, 0].
        If rot is None, defaults to identity rotation (no rotation).
        """
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
    
    def _from_dual_quaternion(self, dq: np.ndarray) -> np.ndarray:
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
    
    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (w, x, y, z format).
        
        Supports both single quaternions (4,) and batch (N, 4).
        """
        if q1.ndim == 1 and q2.ndim == 1:
            # Single quaternion multiplication
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
        else:
            # Batch quaternion multiplication (N, 4) × (N, 4) -> (N, 4)
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            return np.column_stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
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
    
    def _normalize_batch_matrices(self, matrices: np.ndarray) -> np.ndarray:
        """
        Vectorized normalization for batch of matrices using SVD.
        
        Args:
            matrices: (N, 4, 4) array of transformation matrices
            
        Returns:
            (N, 4, 4) array of normalized matrices
        """
        matrices = np.asarray(matrices, dtype=float)
        N = matrices.shape[0]
        
        # Extract rotation parts: (N, 3, 3)
        R_batch = matrices[:, :3, :3]
        t_batch = matrices[:, :3, 3]
        
        # Batch SVD normalization
        # Check orthogonality for each matrix
        R_RT = R_batch @ R_batch.transpose(0, 2, 1)  # (N, 3, 3)
        I_3 = np.eye(3)[np.newaxis, :, :]  # (1, 3, 3)
        orthogonality_errors = np.linalg.norm(R_RT - I_3, axis=(1, 2))
        
        # Use vectorized SVD for batch processing
        U, _, Vt = np.linalg.svd(R_batch)  # All outputs are (N, 3, 3)
        R_normalized = U @ Vt
        
        # Fix reflections (det < 0)
        dets = np.linalg.det(R_normalized)
        need_fix = dets < 0
        if np.any(need_fix):
            U[need_fix, :, -1] *= -1
            R_normalized[need_fix] = U[need_fix] @ Vt[need_fix]
        
        # Warn if significant corrections were made
        if np.any(orthogonality_errors > 1e-6):
            max_error = orthogonality_errors.max()
            warnings.warn(f"Some rotation matrices were not orthogonal (max error: {max_error:.2e}), "
                         "normalized using SVD.", UserWarning)
        
        # Reconstruct batch of 4×4 matrices
        result = np.tile(np.eye(4), (N, 1, 1))
        result[:, :3, :3] = R_normalized
        result[:, :3, 3] = t_batch
        return result
    
    def _normalize_rotation_matrix(self, R_input: np.ndarray) -> np.ndarray:
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
    
    def _exp_se3(self, se3: np.ndarray) -> np.ndarray:
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
    
    def _log_se3(self, matrix: np.ndarray) -> np.ndarray:
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
            theta = np.linalg.norm(omega)
            omega_hat = self._skew_symmetric(omega)  # Use unnormalized omega
            
            # Inverse of left Jacobian using standard formula
            # V_inv = I - 1/2 * [omega]× + (1/θ² - (1+cos(θ))/(2θsin(θ))) * [omega]×²
            half_theta = theta / 2
            
            # Compute the coefficient carefully to avoid numerical issues
            if theta < 1e-4:
                # Taylor expansion for small angles
                coeff = 1.0 / 12.0
            else:
                # Standard formula: (1 - (θ/2) * cot(θ/2)) / θ²
                tan_half = np.tan(half_theta)
                if abs(tan_half) < 1e-10:
                    # Near π multiples
                    coeff = 1.0 / 12.0  # Use approximation
                else:
                    coeff = (1.0 / theta - 1.0 / (2.0 * tan_half)) / theta
            
            V_inv = (np.eye(3) - 
                     0.5 * omega_hat + 
                     coeff * (omega_hat @ omega_hat))
            
            v = V_inv @ t
        
        return np.concatenate([v, omega])
    
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from 3D vector(s).
        
        Supports both single vector (3,) and batch (N, 3).
        """
        if v.ndim == 1:
            # Single vector
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        else:
            # Batch of vectors (N, 3) -> (N, 3, 3)
            N = v.shape[0]
            skew = np.zeros((N, 3, 3))
            skew[:, 0, 1] = -v[:, 2]
            skew[:, 0, 2] = v[:, 1]
            skew[:, 1, 0] = v[:, 2]
            skew[:, 1, 2] = -v[:, 0]
            skew[:, 2, 0] = -v[:, 1]
            skew[:, 2, 1] = v[:, 0]
            return skew
    
    def _compute_all_representations(self) -> None:
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
    def matrix(self) -> np.ndarray:
        """Get 4×4 homogeneous transformation matrix."""
        return self._matrix
    
    @property
    def pos(self) -> np.ndarray:
        """Get translation vector (3D)."""
        if 'pos' not in self._cache:
            if self._is_batch:
                self._cache['pos'] = self._matrix[:, :3, 3]
            else:
                self._cache['pos'] = self._matrix[:3, 3]
        return self._cache['pos']
    
    @property
    def rot(self) -> R:
        """Get rotation as SciPy Rotation object."""
        if 'rot' not in self._cache:
            if self._is_batch:
                self._cache['rot'] = R.from_matrix(self._matrix[:, :3, :3])
            else:
                self._cache['rot'] = R.from_matrix(self._matrix[:3, :3])
        return self._cache['rot']
    
    @property
    def quat(self) -> np.ndarray:
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
    def rotvec(self) -> np.ndarray:
        """Get rotation as rotation vector (axis-angle)."""
        if 'rotvec' not in self._cache:
            self._cache['rotvec'] = self.rot.as_rotvec()
        return self._cache['rotvec']
    
    @property
    def se3(self) -> np.ndarray:
        """Get pose as 6D exponential coordinates (se(3) Lie algebra)."""
        if 'se3' not in self._cache:
            if self._is_batch:
                self._cache['se3'] = np.array([self._log_se3(m) for m in self._matrix])
            else:
                self._cache['se3'] = self._log_se3(self._matrix)
        return self._cache['se3']
    
    @property
    def adjoint(self) -> np.ndarray:
        """
        Get 6×6 adjoint matrix.
        
        The adjoint representation is used for transforming se(3) vectors.
        """
        if 'adjoint' not in self._cache:
            if self._is_batch:
                n = self._batch_size
                R_batch = self._matrix[:, :3, :3]  # (N, 3, 3)
                t_batch = self._matrix[:, :3, 3]   # (N, 3)
                
                # Vectorized skew-symmetric
                t_hat = self._skew_symmetric(t_batch)  # (N, 3, 3)
                
                # Vectorized adjoint construction
                adj = np.zeros((n, 6, 6))
                adj[:, :3, :3] = R_batch
                adj[:, :3, 3:] = t_hat @ R_batch  # (N, 3, 3) @ (N, 3, 3)
                adj[:, 3:, 3:] = R_batch
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
    def screw_axis(self) -> np.ndarray:
        """Get screw axis (unit vector)."""
        if 'screw_axis' not in self._cache:
            omega = self.rotvec
            if self._is_batch:
                theta = np.linalg.norm(omega, axis=1, keepdims=True)
                # For pure translation, use position vector direction
                v = self.pos
                v_norm = np.linalg.norm(v, axis=1, keepdims=True)
                
                # Batch computation: use omega/theta for rotation, v/v_norm for pure translation
                axis_rotation = omega / np.where(theta < 1e-8, 1.0, theta)
                axis_translation = v / np.where(v_norm < 1e-8, 1.0, v_norm)
                
                # Select based on whether theta is small (pure translation)
                is_pure_translation = (theta < 1e-8).squeeze()
                axis = np.where(is_pure_translation[:, np.newaxis], axis_translation, axis_rotation)
                
                # Default to [0, 0, 1] if both theta and v_norm are tiny
                default_axis = np.array([0, 0, 1])
                is_degenerate = ((theta < 1e-8) & (v_norm < 1e-8)).squeeze()
                axis = np.where(is_degenerate[:, np.newaxis], default_axis, axis)
                
                self._cache['screw_axis'] = axis
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
    def screw_pitch(self) -> np.ndarray | float:
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
    def screw_point(self) -> np.ndarray:
        """Get a point on the screw axis."""
        if 'screw_point' not in self._cache:
            omega = self.rotvec
            v = self.pos
            
            if self._is_batch:
                theta = np.linalg.norm(omega, axis=1, keepdims=True)
                # For pure translation, return origin
                # theta[:, 0:1] preserves (N, 1) shape for broadcasting in np.where
                theta_magnitude = theta[:, 0:1]  # Extract for clarity
                point = np.where(theta_magnitude < 1e-8, 
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
    def dual_quat(self) -> np.ndarray:
        """Get dual quaternion representation (8D)."""
        if 'dual_quat' not in self._cache:
            qr = self.quat  # Real part (rotation quaternion)
            t = self.pos    # Translation
            
            if self._is_batch:
                # Dual part: qd = 0.5 * t_quat * qr (vectorized)
                t_quat = np.zeros((self._batch_size, 4))
                t_quat[:, 1:] = t
                
                # Vectorized quaternion multiplication
                qd = 0.5 * self._quat_multiply(t_quat, qr)
                
                self._cache['dual_quat'] = np.hstack([qr, qd])
            else:
                # Dual part: qd = 0.5 * t_quat * qr
                t_quat = np.array([0, t[0], t[1], t[2]])
                qd = 0.5 * self._quat_multiply(t_quat, qr)
                
                self._cache['dual_quat'] = np.concatenate([qr, qd])
        return self._cache['dual_quat']
    
    # Methods
    
    def clear_cache(self) -> None:
        """
        Clear the internal cache of computed representations.
        
        This can be useful for memory management when working with many poses,
        or after modifying the internal matrix directly (which should generally be avoided).
        The cache will be automatically repopulated on next access to properties.
        """
        self._cache.clear()
    
    def normalize(self) -> 'Pose6DOF':
        """
        Normalize the pose to correct numerical drift.
        
        Uses Gram-Schmidt orthogonalization for efficiency (O(n²)).
        This is lighter than SVD and suitable for correcting small numerical errors.
        """
        # Extract current rotation and translation
        if self._is_batch:
            # Vectorized Gram-Schmidt orthogonalization for batch
            R_mat = self._matrix[:, :3, :3]  # (N, 3, 3)
            
            # First column
            r1 = R_mat[:, :, 0]  # (N, 3)
            r1 = r1 / np.linalg.norm(r1, axis=1, keepdims=True)  # (N, 3)
            
            # Second column
            r2 = R_mat[:, :, 1]  # (N, 3)
            r2 = r2 - np.einsum('ij,ij->i', r2, r1)[:, np.newaxis] * r1  # (N, 3)
            r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)  # (N, 3)
            
            # Third column (cross product)
            r3 = np.cross(r1, r2)  # (N, 3)
            
            # Stack columns
            self._matrix[:, :3, :3] = np.stack([r1, r2, r3], axis=2)  # (N, 3, 3)
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
    
    def inverse(self) -> 'Pose6DOF':
        """
        Compute inverse transformation using geometric inversion.
        
        Uses R^T and -R^T * t for efficiency, avoiding matrix inverse.
        Time complexity: O(1)
        """
        if self._is_batch:
            # Vectorized batch inversion
            R_inv = self._matrix[:, :3, :3].transpose(0, 2, 1)  # (N, 3, 3)
            t = self._matrix[:, :3, 3]  # (N, 3)
            # Vectorized matrix-vector multiplication: (N, 3, 3) @ (N, 3) -> (N, 3)
            t_inv = -np.einsum('nij,nj->ni', R_inv, t)  # (N, 3)
            
            # Construct inverse matrices
            inv_matrices = np.tile(np.eye(4), (self._batch_size, 1, 1))
            inv_matrices[:, :3, :3] = R_inv
            inv_matrices[:, :3, 3] = t_inv
            
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
    
    def between(self, other: 'Pose6DOF') -> np.ndarray:
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
    
    def __mul__(self, other: 'Pose6DOF | np.ndarray | list') -> 'Pose6DOF | np.ndarray':
        """
        Multiplication operator for pose composition and point transformation.
        
        Supports:
        - Pose * Pose: Composition of transformations (including Batch×Single, Single×Batch, Batch×Batch)
        - Pose * Vector: Transform 3D point
        - Pose * Vectors: Transform multiple 3D points
        """
        if isinstance(other, Pose6DOF):
            # Pose composition with broadcasting support
            if self._is_batch and other._is_batch:
                # Batch × Batch
                if self._batch_size != other._batch_size:
                    raise ValueError(f"Batch sizes must match: {self._batch_size} vs {other._batch_size}")
                result_matrix = self._matrix @ other._matrix
                result_is_batch = True
                result_batch_size = self._batch_size
            elif self._is_batch and not other._is_batch:
                # Batch × Single - broadcast single to all batch elements
                result_matrix = self._matrix @ other._matrix[np.newaxis, :, :]
                result_is_batch = True
                result_batch_size = self._batch_size
            elif not self._is_batch and other._is_batch:
                # Single × Batch - broadcast single to all batch elements
                result_matrix = self._matrix[np.newaxis, :, :] @ other._matrix
                result_is_batch = True
                result_batch_size = other._batch_size
            else:
                # Single × Single
                result_matrix = self._matrix @ other._matrix
                result_is_batch = False
                result_batch_size = 0
            
            result = Pose6DOF.__new__(Pose6DOF)
            result._matrix = result_matrix
            result._cache = {}
            result._is_batch = result_is_batch
            result._batch_size = result_batch_size
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
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_batch:
            return f"Pose6DOF(batch_size={self._batch_size})"
        else:
            pos = self.pos
            rotvec = self.rotvec
            return f"Pose6DOF(pos={pos}, rotvec={rotvec})"
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Pose6DOF):
            return False
        return np.allclose(self._matrix, other._matrix)
