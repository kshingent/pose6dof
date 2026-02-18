# pose6dof

A NumPy/SciPy-based SE(3) utility class for 6-DOF poses with hybrid lazy/eager evaluation.

## Features

- **Flexible Initialization**: Create poses from translation, rotation matrix, quaternion, Euler angles, axis-angle, 4x4 matrix, or se(3) vector
- **Hybrid Evaluation**: Choose between lazy evaluation (compute on demand) or eager evaluation (pre-compute all representations)
- **Multiple Representations**: Access matrix, quaternion, Euler angles, and se(3) representations seamlessly
- **Operator Overloading**: Intuitive `*` operator for pose composition and point transformation
- **Numerical Stability**: Built-in `.normalize()` method to ensure orthonormal rotations
- **Inverse Transforms**: Compute inverse poses easily
- **Type Safety**: Full type hints and input validation

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from pose6dof import Pose6D

# Create a pose from translation and Euler angles
pose = Pose6D(translation=[1, 2, 3], euler=[0, 0, np.pi/4])

# Access different representations
print(pose.translation)      # [1. 2. 3.]
print(pose.quat)             # Quaternion [x, y, z, w]
print(pose.matrix)           # 4x4 homogeneous matrix
print(pose.euler)            # Euler angles [roll, pitch, yaw]

# Compose poses
pose1 = Pose6D(translation=[1, 0, 0])
pose2 = Pose6D(translation=[0, 1, 0])
pose3 = pose1 * pose2

# Transform points
point = np.array([1, 0, 0])
transformed = pose * point

# Get inverse
pose_inv = pose.inverse()
```

## Usage Examples

### 1. Different Initialization Methods

```python
# Identity pose
pose = Pose6D.identity()

# From translation only
pose = Pose6D(translation=[1, 2, 3])

# From rotation matrix
rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
pose = Pose6D(rotation=rotation)

# From quaternion [x, y, z, w]
pose = Pose6D(quaternion=[0, 0, 0.7071, 0.7071])

# From Euler angles [roll, pitch, yaw]
pose = Pose6D(euler=[0.1, 0.2, 0.3])

# From 4x4 homogeneous matrix
matrix = np.eye(4)
matrix[:3, 3] = [1, 2, 3]
pose = Pose6D(matrix=matrix)

# From se(3) vector [rotation_vector(3), translation(3)]
pose = Pose6D.from_se3([0.1, 0.2, 0.3, 1, 2, 3])
```

### 2. Lazy vs Eager Evaluation

```python
# Lazy evaluation (default) - compute representations on demand
pose_lazy = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
# Properties are computed only when accessed

# Eager evaluation - pre-compute all representations
pose_eager = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3], eager=True)
# All representations (matrix, quat, euler, se3) are computed immediately
```

### 3. Pose Composition

```python
pose1 = Pose6D(translation=[1, 0, 0])
pose2 = Pose6D(translation=[0, 1, 0], euler=[0, 0, np.pi/2])

# Compose: pose3 = pose1 * pose2
pose3 = pose1 * pose2
```

### 4. Point Transformation

```python
pose = Pose6D(translation=[2, 3, 4], euler=[0, 0, np.pi/2])

# Transform a single point
point = np.array([1, 0, 0])
transformed = pose * point

# Transform multiple points
points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
transformed_points = pose * points
```

### 5. Normalization

```python
pose = Pose6D(translation=[1, 2, 3], euler=[0.1, 0.2, 0.3])
normalized = pose.normalize()  # Ensures rotation is orthonormal
```

## API Reference

### Class: `Pose6D`

#### Initialization Parameters

- `translation` (array-like, optional): 3D translation vector
- `rotation` (array-like or Rotation, optional): 3x3 rotation matrix or scipy Rotation object
- `quaternion` (array-like, optional): Quaternion [x, y, z, w]
- `euler` (array-like, optional): Euler angles [roll, pitch, yaw] in radians
- `axis_angle` (array-like, optional): Axis-angle representation (3D vector)
- `matrix` (array-like, optional): 4x4 homogeneous transformation matrix
- `eager` (bool, optional): If True, pre-compute all representations. Default: False

#### Properties

- `translation`: Get the translation vector (3D array)
- `rotation_matrix`: Get the 3x3 rotation matrix
- `matrix`: Get the 4x4 homogeneous transformation matrix (cached)
- `quat`: Get the quaternion [x, y, z, w] (cached)
- `euler`: Get Euler angles [roll, pitch, yaw] in radians (cached)
- `se3`: Get the se(3) Lie algebra representation (cached)

#### Methods

- `normalize()`: Return a new Pose6D with normalized rotation
- `inverse()`: Return the inverse transformation
- `to_dict()`: Convert to dictionary with all representations
- `identity(eager=False)`: Class method to create identity pose
- `from_se3(se3_vector, eager=False)`: Class method to create from se(3) vector

#### Operators

- `pose1 * pose2`: Compose two poses
- `pose * point`: Transform a point or array of points

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Examples

See the `examples/` directory for more detailed usage examples:

```bash
python examples/usage_example.py
```

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

MIT License - see LICENSE file for details
