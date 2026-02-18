"""
pose6dof.core
=============

SE(3) pose representation with hybrid lazy/eager evaluation.

Core Concept:
-------------
The Pose6DOF class represents a 6DOF rigid body transformation using a 4×4 
homogeneous transformation matrix as its internal representation (self._matrix).

The class uses a hybrid evaluation system combining:
- Lazy Evaluation: Expensive computations are deferred until accessed
- Caching: Once computed, representations are cached for O(1) access
- Eager Mode: Optional pre-computation of all representations
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple, Optional, Dict, Any
import warnings


class Pose6DOF:
    """
    SE(3) pose representation with 4×4 matrix internal representation.
    
    Internal Representation:
    ------------------------
    The internal representation is stored as a 4×4 homogeneous transformation 
    matrix in self._matrix:
    
        ┌         ┐
        │ R  |  t │  (3×3 rotation matrix R, 3×1 translation t)
        │----+----│
        │ 0  |  1 │  (bottom row is [0, 0, 0, 1])
        └         ┘
    
    Input Formats:
    --------------
    The constructor supports various input formats:
    
    1. **No input**: Identity matrix (default)
       >>> p = Pose6DOF()
    
    2. **4×4 Affine Matrix**: Homogeneous transformation matrix
       >>> p = Pose6DOF(np.eye(4))
    
    3. **6D Exponential Coordinates**: se(3) Lie algebra
       >>> p = Pose6DOF(np.array([0.1, 0, 0, 0, 0, 0]))  # twist
    
    4. **Separated (pos, rot)**: Translation and rotation
       >>> p = Pose6DOF(pos=[1, 2, 3], rot=R.from_euler('xyz', [90, 0, 0], degrees=True))
    
    5. **Pose6DOF object**: Copy constructor
       >>> p2 = Pose6DOF(p)
    
    6. **Dual Quaternions**: [real_quat(4), dual_quat(4)]
       >>> p = Pose6DOF(dual_quat=np.array([1, 0, 0, 0, 0, 0.5, 0, 0]))
    
    Rotation Formats (for rot parameter):
    --------------------------------------
    - 3×3 rotation matrix
    - 4D quaternion [w, x, y, z]
    - 3D rotation vector (axis-angle)
    - SciPy Rotation object
    
    Special Input Methods:
    ----------------------
    - from_screw_param(): Create from screw parameters (not auto-detected)
    - from_euler(): Create from Euler angles (not auto-detected)
    
    Batch Operations:
    -----------------
    Supports batch processing with NumPy broadcasting for (N, 4, 4) arrays:
    >>> poses = Pose6DOF(np.random.randn(10, 4, 4))  # 10 poses at once
    
    Note: The normalize() method converts _matrix→R→_matrix via scipy's Rotation
    to eliminate accumulated numerical drift. Computational cost: O(n³) for 
    3×3 SVD decomposition.
    
    Advanced Features:
    ------------------
    - Relative Pose: Extract relative pose ΔT = T₁⁻¹T₂ as Lie algebra (6D vector)
    - Interpolation: SE(3) interpolation preserving Lie group structure using 
      exponential mapping (not simple linear interpolation)
    
    Parameters
    ----------
    *args : various
        Positional arguments for flexible initialization
    pos : array-like, optional
        Translation vector (3,)
    rot : array-like or Rotation, optional
        Rotation in various formats
    dual_quat : array-like, optional
        Dual quaternion [real(4), dual(4)]
    eager : bool, default=False
        If True, compute all representations immediately
    **kwargs : dict
        Additional keyword arguments
    """
    
    def __init__(
        self,
        *args,
        pos: Optional[np.ndarray] = None,
        rot: Optional[Union[np.ndarray, R]] = None,
        dual_quat: Optional[np.ndarray] = None,
        eager: bool = False,
        **kwargs
    ):
        """Initialize Pose6DOF from various input formats."""
        
        # Internal representation: 4×4 homogeneous transformation matrix
        self._matrix: np.ndarray = None
        
        # Cache for lazy evaluation
        self._cache: Dict[str, Any] = {}
        self._eager = eager
        
        # Handle dual quaternion input
        if dual_quat is not None:
            self._matrix = self._dual_quat_to_matrix(dual_quat)
        # Handle no input → identity matrix (default)
        elif len(args) == 0 and pos is None and rot is None:
            self._matrix = np.eye(4)
        # Handle Pose6DOF object (copy constructor)
        elif len(args) == 1 and isinstance(args[0], Pose6DOF):
            self._matrix = args[0]._matrix.copy()
        # Handle single array input
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            arr = np.asarray(args[0])
            if arr.shape == (4, 4) or (arr.ndim == 3 and arr.shape[1:] == (4, 4)):
                # 4×4 matrix or batch of matrices
                self._matrix = arr.copy()
            elif arr.shape == (6,) or (arr.ndim == 2 and arr.shape[1] == 6):
                # 6D exponential coordinates (se(3))
                self._matrix = self._exp_map(arr)
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
        # Handle separated pos and rot
        elif pos is not None or rot is not None:
            pos_vec = np.zeros(3) if pos is None else np.asarray(pos)
            rot_mat = self._parse_rotation(rot) if rot is not None else np.eye(3)
            self._matrix = self._build_matrix(rot_mat, pos_vec)
        # Handle two positional arguments (pos, rot)
        elif len(args) == 2:
            pos_vec = np.asarray(args[0])
            rot_mat = self._parse_rotation(args[1])
            self._matrix = self._build_matrix(rot_mat, pos_vec)
        else:
            raise ValueError(f"Unsupported initialization arguments")
        
        # Validate matrix shape
        if self._matrix.ndim == 3 and self._matrix.shape[1:] == (4, 4):
            pass  # Batch mode (N, 4, 4)
        elif self._matrix.shape == (4, 4):
            pass  # Single pose (4, 4)
        else:
            raise ValueError(f"Invalid matrix shape: {self._matrix.shape}")
        
        # Normalize rotation part if needed
        if self._matrix.ndim == 2:
            self._normalize_rotation_inplace()
        
        # Pre-compute all representations if eager mode
        if eager:
            self._compute_all()
    
    @classmethod
    def from_screw_param(
        cls,
        axis: np.ndarray,
        point: np.ndarray,
        pitch: float,
        theta: float
    ) -> 'Pose6DOF':
        """
        Create Pose6DOF from screw parameters.
        
        This is NOT auto-detected and must be called explicitly.
        
        Parameters
        ----------
        axis : array-like, shape (3,)
            Screw axis (unit vector)
        point : array-like, shape (3,)
            A point on the screw axis
        pitch : float
            Screw pitch (translation per radian of rotation)
        theta : float
            Rotation angle in radians
        
        Returns
        -------
        Pose6DOF
            Pose constructed from screw parameters
        """
        axis = np.asarray(axis)
        point = np.asarray(point)
        
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        # Compute moment: m = p × s
        moment = np.cross(point, axis)
        
        # Build 6D twist (angular velocity, linear velocity)
        omega = axis * theta
        v = (pitch * theta * axis + np.cross(moment, omega))
        
        # Create se(3) vector [v, omega]
        twist = np.concatenate([v, omega])
        
        return cls(twist)
    
    @classmethod
    def from_euler(
        cls,
        angles: Union[float, np.ndarray],
        seq: str,
        degrees: bool = False,
        pos: Optional[np.ndarray] = None
    ) -> 'Pose6DOF':
        """
        Create Pose6DOF from Euler angles.
        
        This is NOT auto-detected and must be called explicitly via from_euler().
        
        Parameters
        ----------
        angles : float or array-like
            Euler angles
        seq : str
            Rotation sequence (e.g., 'xyz', 'zyx')
        degrees : bool, default=False
            If True, angles are in degrees
        pos : array-like, optional
            Translation vector
        
        Returns
        -------
        Pose6DOF
            Pose constructed from Euler angles
        """
        rot = R.from_euler(seq, angles, degrees=degrees)
        pos_vec = np.zeros(3) if pos is None else np.asarray(pos)
        return cls(pos=pos_vec, rot=rot)
    
    @staticmethod
    def _parse_rotation(rot: Union[np.ndarray, R, Dict]) -> np.ndarray:
        """Parse rotation from various formats to 3×3 matrix."""
        if rot is None:
            return np.eye(3)
        elif isinstance(rot, R):
            return rot.as_matrix()
        elif isinstance(rot, dict):
            # Dictionary format for Euler angles (legacy support, prefer from_euler)
            warnings.warn(
                "Dictionary format for Euler angles is deprecated. "
                "Use Pose6DOF.from_euler() instead.",
                DeprecationWarning,
                stacklevel=3
            )
            seq = rot.get('seq', 'xyz')
            angles = rot.get('angles', [0, 0, 0])
            degrees = rot.get('degrees', False)
            return R.from_euler(seq, angles, degrees=degrees).as_matrix()
        else:
            rot = np.asarray(rot)
            if rot.shape == (3, 3):
                # 3×3 rotation matrix
                return rot
            elif rot.shape == (4,):
                # Quaternion [w, x, y, z]
                return R.from_quat([rot[1], rot[2], rot[3], rot[0]]).as_matrix()
            elif rot.shape == (3,):
                # Rotation vector (axis-angle)
                return R.from_rotvec(rot).as_matrix()
            else:
                raise ValueError(f"Unsupported rotation format: {rot.shape}")
    
    @staticmethod
    def _build_matrix(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Build 4×4 homogeneous transformation matrix."""
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        return mat
    
    @staticmethod
    def _dual_quat_to_matrix(dq: np.ndarray) -> np.ndarray:
        """
        Convert dual quaternion to 4×4 matrix.
        
        Dual quaternion format: [q_real(4), q_dual(4)]
        where q_real = [w, x, y, z] for rotation
        and q_dual encodes translation
        """
        dq = np.asarray(dq)
        if dq.shape != (8,):
            raise ValueError(f"Dual quaternion must be shape (8,), got {dq.shape}")
        
        # Extract real and dual parts
        q_r = dq[:4]  # [w, x, y, z]
        q_d = dq[4:]  # dual part
        
        # Normalize real quaternion
        q_r = q_r / np.linalg.norm(q_r)
        
        # Convert real quaternion to rotation matrix
        rot = R.from_quat([q_r[1], q_r[2], q_r[3], q_r[0]]).as_matrix()
        
        # Extract translation: t = 2 * q_d * q_r^*
        # q_r^* is conjugate: [w, -x, -y, -z]
        q_r_conj = np.array([q_r[0], -q_r[1], -q_r[2], -q_r[3]])
        
        # Quaternion multiplication: t = 2 * q_d * q_r_conj
        t_quat = 2 * np.array([
            q_d[0] * q_r_conj[0] - q_d[1] * q_r_conj[1] - q_d[2] * q_r_conj[2] - q_d[3] * q_r_conj[3],
            q_d[0] * q_r_conj[1] + q_d[1] * q_r_conj[0] + q_d[2] * q_r_conj[3] - q_d[3] * q_r_conj[2],
            q_d[0] * q_r_conj[2] - q_d[1] * q_r_conj[3] + q_d[2] * q_r_conj[0] + q_d[3] * q_r_conj[1],
            q_d[0] * q_r_conj[3] + q_d[1] * q_r_conj[2] - q_d[2] * q_r_conj[1] + q_d[3] * q_r_conj[0]
        ])
        
        # Translation is the vector part
        pos = t_quat[1:]
        
        return Pose6DOF._build_matrix(rot, pos)
    
    @staticmethod
    def _exp_map(twist: np.ndarray) -> np.ndarray:
        """
        Exponential map from se(3) to SE(3).
        
        Converts 6D twist vector to 4×4 transformation matrix.
        twist = [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        """
        twist = np.asarray(twist)
        
        # Handle batch mode
        if twist.ndim == 2:
            return np.array([Pose6DOF._exp_map(t) for t in twist])
        
        v = twist[:3]  # linear velocity
        omega = twist[3:]  # angular velocity
        
        theta = np.linalg.norm(omega)
        
        if theta < 1e-8:
            # Small angle: use Taylor expansion
            R_mat = np.eye(3) + Pose6DOF._skew(omega)
            t = v
        else:
            # Rodrigues' formula
            omega_hat = Pose6DOF._skew(omega)
            omega_unit = omega / theta
            omega_unit_hat = Pose6DOF._skew(omega_unit)
            
            R_mat = (np.eye(3) + 
                    np.sin(theta) * omega_unit_hat + 
                    (1 - np.cos(theta)) * omega_unit_hat @ omega_unit_hat)
            
            # Translation part
            V = (np.eye(3) + 
                 (1 - np.cos(theta)) / theta * omega_unit_hat + 
                 (theta - np.sin(theta)) / theta * omega_unit_hat @ omega_unit_hat)
            t = V @ v
        
        return Pose6DOF._build_matrix(R_mat, t)
    
    @staticmethod
    def _log_map(matrix: np.ndarray) -> np.ndarray:
        """
        Logarithm map from SE(3) to se(3).
        
        Converts 4×4 transformation matrix to 6D twist vector.
        """
        R_mat = matrix[:3, :3]
        t = matrix[:3, 3]
        
        # Extract rotation angle
        trace = np.trace(R_mat)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if theta < 1e-8:
            # Small angle
            omega = np.array([
                R_mat[2, 1] - R_mat[1, 2],
                R_mat[0, 2] - R_mat[2, 0],
                R_mat[1, 0] - R_mat[0, 1]
            ]) / 2
            v = t
        else:
            # Extract axis
            omega_hat = (R_mat - R_mat.T) / (2 * np.sin(theta))
            omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]) * theta
            
            omega_unit = omega / theta
            omega_unit_hat = Pose6DOF._skew(omega_unit)
            
            # Inverse of V
            V_inv = (np.eye(3) - 
                    0.5 * theta * omega_unit_hat + 
                    (1 - theta / (2 * np.tan(theta / 2))) * omega_unit_hat @ omega_unit_hat)
            v = V_inv @ t
        
        return np.concatenate([v, omega])
    
    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from 3D vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def _normalize_rotation_inplace(self):
        """Normalize rotation matrix using SVD."""
        if self._matrix.ndim == 2:
            R_mat = self._matrix[:3, :3]
            U, _, Vt = np.linalg.svd(R_mat)
            R_normalized = U @ Vt
            # Ensure det(R) = 1 (proper rotation)
            if np.linalg.det(R_normalized) < 0:
                Vt[-1, :] *= -1
                R_normalized = U @ Vt
            self._matrix[:3, :3] = R_normalized
    
    def normalize(self) -> 'Pose6DOF':
        """
        Normalize the pose to eliminate numerical drift.
        
        This method converts the internal 4×4 matrix to scipy's Rotation object
        and back: _matrix → R → _matrix. This eliminates accumulated floating-point
        errors from repeated operations.
        
        Computational Cost:
        -------------------
        O(n³) for 3×3 SVD decomposition in scipy's Rotation.as_matrix()
        
        The normalization uses Singular Value Decomposition (SVD) to reconstruct
        the rotation matrix, which is more stable than Gram-Schmidt and prevents
        reflections (det(R) = -1) while recovering the closest proper rotation.
        
        Returns
        -------
        self : Pose6DOF
            Returns self for method chaining
        """
        if self._matrix.ndim == 2:
            # Extract rotation and translation
            rot_mat = self._matrix[:3, :3]
            pos = self._matrix[:3, 3]
            
            # Convert through scipy's R: _matrix → R → _matrix
            rot = R.from_matrix(rot_mat)
            rot_normalized = rot.as_matrix()
            
            # Rebuild matrix
            self._matrix[:3, :3] = rot_normalized
            self._matrix[:3, 3] = pos
            
            # Clear cache since matrix changed
            self._cache.clear()
        else:
            # Batch mode
            for i in range(self._matrix.shape[0]):
                rot_mat = self._matrix[i, :3, :3]
                pos = self._matrix[i, :3, 3]
                
                rot = R.from_matrix(rot_mat)
                rot_normalized = rot.as_matrix()
                
                self._matrix[i, :3, :3] = rot_normalized
                self._matrix[i, :3, 3] = pos
        
        return self
    
    def _compute_all(self):
        """Pre-compute all representations (eager mode)."""
        _ = self.matrix
        _ = self.pos
        _ = self.rot
        _ = self.quat
        _ = self.rotvec
        _ = self.se3
        _ = self.adjoint
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4×4 homogeneous transformation matrix."""
        return self._matrix.copy()
    
    @property
    def pos(self) -> np.ndarray:
        """Get translation vector (3,)."""
        if 'pos' not in self._cache:
            self._cache['pos'] = self._matrix[:3, 3].copy()
        return self._cache['pos']
    
    @property
    def rot(self) -> R:
        """Get rotation as scipy Rotation object."""
        if 'rot' not in self._cache:
            if self._matrix.ndim == 2:
                self._cache['rot'] = R.from_matrix(self._matrix[:3, :3])
            else:
                # Batch mode
                self._cache['rot'] = R.from_matrix(self._matrix[:, :3, :3])
        return self._cache['rot']
    
    @property
    def quat(self) -> np.ndarray:
        """Get quaternion [w, x, y, z]."""
        if 'quat' not in self._cache:
            q = self.rot.as_quat()  # scipy format: [x, y, z, w]
            # Convert to [w, x, y, z]
            if q.ndim == 1:
                self._cache['quat'] = np.array([q[3], q[0], q[1], q[2]])
            else:
                # Batch mode
                self._cache['quat'] = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
        return self._cache['quat']
    
    @property
    def rotvec(self) -> np.ndarray:
        """Get rotation vector (axis-angle representation)."""
        if 'rotvec' not in self._cache:
            self._cache['rotvec'] = self.rot.as_rotvec()
        return self._cache['rotvec']
    
    @property
    def se3(self) -> np.ndarray:
        """Get se(3) Lie algebra representation (6D twist)."""
        if 'se3' not in self._cache:
            if self._matrix.ndim == 2:
                self._cache['se3'] = self._log_map(self._matrix)
            else:
                # Batch mode
                self._cache['se3'] = np.array([self._log_map(m) for m in self._matrix])
        return self._cache['se3']
    
    @property
    def adjoint(self) -> np.ndarray:
        """
        Get 6×6 adjoint matrix representation.
        
        The adjoint matrix Ad_T transforms twists from one frame to another.
        """
        if 'adjoint' not in self._cache:
            if self._matrix.ndim == 2:
                R_mat = self._matrix[:3, :3]
                t = self._matrix[:3, 3]
                t_skew = self._skew(t)
                
                adj = np.zeros((6, 6))
                adj[:3, :3] = R_mat
                adj[:3, 3:] = t_skew @ R_mat
                adj[3:, 3:] = R_mat
                
                self._cache['adjoint'] = adj
            else:
                # Batch mode
                adj_list = []
                for i in range(self._matrix.shape[0]):
                    R_mat = self._matrix[i, :3, :3]
                    t = self._matrix[i, :3, 3]
                    t_skew = self._skew(t)
                    
                    adj = np.zeros((6, 6))
                    adj[:3, :3] = R_mat
                    adj[:3, 3:] = t_skew @ R_mat
                    adj[3:, 3:] = R_mat
                    adj_list.append(adj)
                
                self._cache['adjoint'] = np.array(adj_list)
        
        return self._cache['adjoint']
    
    @property
    def dual_quat(self) -> np.ndarray:
        """
        Get dual quaternion representation [q_real(4), q_dual(4)].
        
        Returns
        -------
        np.ndarray, shape (8,)
            Dual quaternion [w, x, y, z, dw, dx, dy, dz]
        """
        if 'dual_quat' not in self._cache:
            q_r = self.quat  # [w, x, y, z]
            t = self.pos
            
            # q_d = 0.5 * t * q_r (quaternion multiplication)
            # where t is treated as quaternion [0, tx, ty, tz]
            t_quat = np.array([0, t[0], t[1], t[2]])
            
            # Quaternion multiplication: q_d = 0.5 * t * q_r
            q_d = 0.5 * np.array([
                t_quat[0] * q_r[0] - t_quat[1] * q_r[1] - t_quat[2] * q_r[2] - t_quat[3] * q_r[3],
                t_quat[0] * q_r[1] + t_quat[1] * q_r[0] + t_quat[2] * q_r[3] - t_quat[3] * q_r[2],
                t_quat[0] * q_r[2] - t_quat[1] * q_r[3] + t_quat[2] * q_r[0] + t_quat[3] * q_r[1],
                t_quat[0] * q_r[3] + t_quat[1] * q_r[2] - t_quat[2] * q_r[1] + t_quat[3] * q_r[0]
            ])
            
            self._cache['dual_quat'] = np.concatenate([q_r, q_d])
        
        return self._cache['dual_quat']
    
    @property
    def screw_axis(self) -> np.ndarray:
        """Get screw axis (unit vector)."""
        if 'screw_axis' not in self._cache:
            omega = self.se3[3:]
            theta = np.linalg.norm(omega)
            if theta < 1e-8:
                self._cache['screw_axis'] = np.array([0, 0, 1])  # arbitrary axis for pure translation
            else:
                self._cache['screw_axis'] = omega / theta
        return self._cache['screw_axis']
    
    @property
    def screw_pitch(self) -> float:
        """Get screw pitch (translation per radian of rotation)."""
        if 'screw_pitch' not in self._cache:
            v = self.se3[:3]
            omega = self.se3[3:]
            theta = np.linalg.norm(omega)
            
            if theta < 1e-8:
                self._cache['screw_pitch'] = np.inf  # pure translation
            else:
                axis = omega / theta
                # Pitch is the component of v along the axis divided by theta
                self._cache['screw_pitch'] = np.dot(v, axis) / theta
        
        return self._cache['screw_pitch']
    
    def inverse(self) -> 'Pose6DOF':
        """
        Compute inverse transformation.
        
        Uses geometric inverse: R^T and -R^T * t instead of matrix inverse.
        
        Returns
        -------
        Pose6DOF
            Inverse pose
        """
        if self._matrix.ndim == 2:
            R_mat = self._matrix[:3, :3]
            t = self._matrix[:3, 3]
            
            R_inv = R_mat.T
            t_inv = -R_inv @ t
            
            mat_inv = self._build_matrix(R_inv, t_inv)
            return Pose6DOF(mat_inv)
        else:
            # Batch mode
            mat_inv_list = []
            for i in range(self._matrix.shape[0]):
                R_mat = self._matrix[i, :3, :3]
                t = self._matrix[i, :3, 3]
                
                R_inv = R_mat.T
                t_inv = -R_inv @ t
                
                mat_inv_list.append(self._build_matrix(R_inv, t_inv))
            
            return Pose6DOF(np.array(mat_inv_list))
    
    def between(self, other: 'Pose6DOF') -> np.ndarray:
        """
        Compute relative pose as Lie algebra (6D vector).
        
        Given two poses T₁ (self) and T₂ (other), computes:
        ΔT = T₁⁻¹ T₂
        
        and returns it as a 6D Lie algebra vector (se(3)).
        
        Parameters
        ----------
        other : Pose6DOF
            Target pose
        
        Returns
        -------
        np.ndarray, shape (6,)
            Relative pose as Lie algebra twist vector [v, ω]
        """
        delta = self.inverse() * other
        return delta.se3
    
    @staticmethod
    def interpolate(
        pose1: 'Pose6DOF',
        pose2: 'Pose6DOF',
        t: Union[float, np.ndarray]
    ) -> 'Pose6DOF':
        """
        Interpolate between two poses on SE(3) using exponential mapping.
        
        This is NOT simple linear interpolation. It preserves the Lie group
        structure by using the exponential map:
        
        T(t) = T₁ * exp(t * log(T₁⁻¹ T₂))
        
        where t ∈ [0, 1], exp is the exponential map, and log is the logarithm map.
        
        Parameters
        ----------
        pose1 : Pose6DOF
            Start pose (at t=0)
        pose2 : Pose6DOF
            End pose (at t=1)
        t : float or array-like
            Interpolation parameter(s) in [0, 1]
        
        Returns
        -------
        Pose6DOF
            Interpolated pose(s)
        """
        # Compute relative pose in Lie algebra
        delta_se3 = pose1.between(pose2)
        
        # Scale by interpolation parameter
        t = np.asarray(t)
        if t.ndim == 0:
            # Single interpolation
            scaled_se3 = t * delta_se3
            delta_T = Pose6DOF(scaled_se3)
            return pose1 * delta_T
        else:
            # Multiple interpolations
            poses = []
            for t_val in t:
                scaled_se3 = t_val * delta_se3
                delta_T = Pose6DOF(scaled_se3)
                poses.append((pose1 * delta_T).matrix)
            return Pose6DOF(np.array(poses))
    
    def __mul__(self, other):
        """
        Compose poses or transform points.
        
        - Pose6DOF * Pose6DOF: Composition T₁ ∘ T₂
        - Pose6DOF * array(3,): Transform 3D point
        - Pose6DOF * array(N, 3): Transform N points
        
        Returns
        -------
        Pose6DOF or np.ndarray
            Composed pose or transformed point(s)
        """
        if isinstance(other, Pose6DOF):
            # Pose composition
            if self._matrix.ndim == 2 and other._matrix.ndim == 2:
                mat_composed = self._matrix @ other._matrix
                return Pose6DOF(mat_composed)
            else:
                # Batch composition
                raise NotImplementedError("Batch pose composition not yet implemented")
        else:
            # Point transformation
            other = np.asarray(other)
            if other.shape == (3,):
                # Single point
                point_h = np.append(other, 1)
                transformed_h = self._matrix @ point_h
                return transformed_h[:3]
            elif other.ndim == 2 and other.shape[1] == 3:
                # Multiple points
                N = other.shape[0]
                points_h = np.column_stack([other, np.ones(N)])
                transformed_h = (self._matrix @ points_h.T).T
                return transformed_h[:, :3]
            else:
                raise ValueError(f"Unsupported shape for point transformation: {other.shape}")
    
    def __repr__(self) -> str:
        """String representation."""
        if self._matrix.ndim == 2:
            return f"Pose6DOF(pos={self.pos}, quat={self.quat})"
        else:
            return f"Pose6DOF(batch_size={self._matrix.shape[0]})"
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Pose6DOF):
            return False
        return np.allclose(self._matrix, other._matrix)
