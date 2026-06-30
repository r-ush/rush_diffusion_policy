from typing import Union
import sys
import torch
import numpy as np
import functools
from scipy.spatial.transform import Rotation as R

# Replacement conversion functions (torch tensor in, torch tensor out)
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_torch(x, ref=None):
    t = torch.from_numpy(np.asarray(x))
    if isinstance(ref, torch.Tensor):
        return t.to(ref.dtype).to(ref.device)
    return t

def axis_angle_to_matrix(x):
    xp = _to_numpy(x)
    shp = xp.shape
    mat = R.from_rotvec(xp.reshape(-1, 3)).as_matrix()
    mat = mat.reshape(*shp[:-1], 3, 3)
    return _to_torch(mat, x)

def matrix_to_axis_angle(mat):
    mp = _to_numpy(mat)
    shp = mp.shape
    rotvec = R.from_matrix(mp.reshape(-1, 3, 3)).as_rotvec()
    rotvec = rotvec.reshape(*shp[:-2], 3)
    return _to_torch(rotvec, mat)

def quaternion_to_matrix(q):
    qp = _to_numpy(q)
    shp = qp.shape
    mat = R.from_quat(qp.reshape(-1, 4)).as_matrix()
    mat = mat.reshape(*shp[:-1], 3, 3)
    return _to_torch(mat, q)

def matrix_to_quaternion(mat):
    mp = _to_numpy(mat)
    shp = mp.shape
    quat = R.from_matrix(mp.reshape(-1, 3, 3)).as_quat()
    quat = quat.reshape(*shp[:-2], 4)
    return _to_torch(quat, mat)

def rotation_6d_to_matrix(x):
    xp = _to_numpy(x)
    a1 = xp[..., :3]
    a2 = xp[..., 3:6]
    # normalize a1
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    # make b2 orthogonal to b1
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    u2 = a2 - proj
    b2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)  # (...,3,3)
    return _to_torch(mat, x)

def matrix_to_rotation_6d(mat):
    mp = _to_numpy(mat)
    # take first two columns
    c1 = mp[..., :, 0]
    c2 = mp[..., :, 1]
    out = np.concatenate([c1, c2], axis=-1)
    return _to_torch(out, mat)

def euler_angles_to_matrix(x, convention='xyz'):
    xp = _to_numpy(x)
    shp = xp.shape
    mat = R.from_euler(convention, xp.reshape(-1, xp.shape[-1])).as_matrix()
    mat = mat.reshape(*shp[:-1], 3, 3)
    return _to_torch(mat, x)

def matrix_to_euler_angles(mat, convention='xyz'):
    mp = _to_numpy(mat)
    shp = mp.shape
    angles = R.from_matrix(mp.reshape(-1, 3, 3)).as_euler(convention)
    angles = angles.reshape(*shp[:-2], angles.shape[-1])
    return _to_torch(angles, mat)

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(sys.modules[__name__], f'{from_rep}_to_matrix'),
                getattr(sys.modules[__name__], f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(sys.modules[__name__], f'matrix_to_{to_rep}'),
                getattr(sys.modules[__name__], f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
