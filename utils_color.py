import numpy as np
import torch

def xyz2xyY(xyz):
    X, Y, Z = xyz
    total = X + Y + Z
    if total == 0:
        return [0, 0]
    x = X / total
    y = Y / total
    return [x, y]

def xyz2sRGB(xyz):
    XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])
    RGB = np.dot(XYZ2RGB, xyz)
    RGB = np.clip(RGB, 0, 1)
    sRGB = lin2sRGB(RGB)
    return sRGB

def xy2XYZ(x, y):
    Y = 1.0  # Set Y to 1.0 (luminance)
    if y == 0:
        return [0, 0, 0]
    X = (x * Y) / y
    Z = (1 - x - y) * Y / y
    return np.array([X, Y, Z])

def lin2sRGB(lin):
    sRGB = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(sRGB, 0, 1)

def f(t):
    delta = 6/29
    if isinstance(t, torch.Tensor):
        result = torch.where(t > delta**3, t**(1/3),t/(3*delta**2) + 4/29)
    else:
        raise RuntimeError('Define for datatypes other than torch.tensor')
    
    return result

def f_inv(t):
    delta = 6/29
    if isinstance(t, torch.Tensor):
        result = torch.where(t > delta, t**3, 3*delta**2*(t-4/29))
    else:
        raise RuntimeError('Define for datatypes other than torch.tensor')
    
    return result


def xyz2lab(xyz, dev, illuminant = 'D65'):
    assert xyz.shape[0] == 3, 'Input should be three-channel format'

    if illuminant =='D65':
        Xn = 95.047
        Yn = 100.000
        Zn = 108.883
    
    lab = torch.zeros_like(xyz).to(dev)
    lab[0,...] = 116 * f(xyz[1,...]/Yn) - 16 
    lab[1,...] = 500 * (f(xyz[0,...]/Xn) - f(xyz[1,...]/Yn))
    lab[2,...] = 200 * (f(xyz[1,...]/Yn) - f(xyz[2,...]/Zn))
    
    return lab

def lab2xyz(lab, dev, illuminant = 'D65'):
    assert lab.shape[0] == 3, 'Input should be three-channel format'


    if illuminant =='D65':
        Xn = 95.047
        Yn = 100.000
        Zn = 108.883

    xyz= torch.zeros_like(lab).to(dev)
    xyz[0,...] = Xn * f_inv((lab[1,...]/500) + (lab[0,...] + 16)/116)
    xyz[1,...] = Yn * f_inv((lab[0,...] + 16)/116)
    xyz[2,...] = Zn * f_inv((lab[0,...] + 16)/116 - (lab[2,...]/200))

    return xyz