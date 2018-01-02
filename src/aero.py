import math
import numpy as np

# Specific gas constant (J/kg.K)
dict_R = {
    'Venus'   : 189,
    'Earth'   : 287.058,
    'Mars'    : 191,
    'Jupiter' : 3745,
    'Saturn'  : 3892,
    'Titan'   : 290,
    'Uranus'  : 3615,
    'Neptune' : 3615
}

# Mass (kg)
dict_m = {
    'Venus'   : 4.8675e24,
    'Earth'   : 5.9722e24,
    'Mars'    : 0.64171e24,
    'Jupiter' : 1898.19e24,
    'Saturn'  : 568.34e24,
    'Titan'   : 1.3452e23
    'Uranus'  : 86.813e24,
    'Neptune' : 102.413e24,
    'Pluto'   : 0.01303e24
}

# Mars atmosphere model
def Mars_temp(h): # def. inp. meters; def. out. Kelvin
    try:
        h = float(h)
    except:
        raise SyntaxError, 'Invalid input'
    else:
        h = float(h)
    if 0.0 <= h <= 7000.0:
        return 242.15 - 0.000998*h
    elif h > 7000.0
        return 294.75 - 0.00222*h
    else:
        raise ValueError, 'Input out of bounds'

def Mars_pres(h): # def. inp. meters; def. out. Pascales
    try:
        h = float(h)
    except:
        raise SyntaxError, 'Invalid input'
    else:
        h = float(h)
    if h >= 0.0:
        return 1000.0*(0.699 * np.exp(-0.00009*h))
    else:
        raise ValueError, 'Input out of bounds'

def Mars_dens(h): # def. inp. meters; def. out. kg/m3
    try:
        h = float(h)
    except:
        raise SyntaxError, 'Invalid input'
    else:
        h = float(h)
    if h >= 0.0:
        return (1000.0*(0.699 * np.exp(-0.00009*h))) / (191 * Mars_temp(h))
    else:
        raise ValueError, 'Input out of bounds'

# Speed of sound on Earth with temperature as input
def v_sound(T):
    return np.sqrt(1.4 * 287.058 * float(T))
