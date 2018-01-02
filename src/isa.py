# ISA Standard Atmosphere 1976
import math
import numpy as np

# Temperature for given altitude (def. inp. meters; def. out. Kelvin)
def temp(h):
	try:
		if (float(h) < -610): 
			raise ValueError, 'Input out of bounds'
			return
	except TypeError:
		raise TypeError, 'Invalid input type'
		return
	if (h >= -610.0 and h <= 11000.0):
		return 288.15 - 0.0065*h
	elif (h > 11000.0 and h <= 20000.0):
		return 216.65
	elif (h > 20000.0 and h <= 32000.0):
		return 216.65 + 0.001*(h-20000.0)
	elif (h > 32000.0 and h <= 47000.0):
		return 228.65 + 0.0028*(h-32000.0)
	elif (h > 47000.0 and h <= 51000.0):
		return 270.65
	elif (h > 51000.0 and h <= 71000.0):
		return 270.65 - 0.0028*(h-51000.0)
	elif (h > 71000.0 and h <= 84855.8345322628):
		return 214.65 - 0.002*(h-71000)
	elif (h > 84855.8345322628):
		return 186.9467

# Density for given altitude (def. inp. meters; def. out. kg/m3)
def dens(h):
	try:
		if (float(h) < -610): 
			raise ValueError, 'Input out of bounds'
			return
	except TypeError:
		raise TypeError, 'Invalid input type'
		return
	if (h >= -610 and h <= 11000.0):
		return 1.225*(temp(h)/288.15)**(-9.80665/(-0.0065*287.058)-1.0)
	elif (h > 11000.0 and h <= 20000.0):
		return 0.36392739145948744*np.exp((-9.80665/(216.65*287.058))*(h - 11000.0))
	elif (h > 20000.0 and h <= 32000.0):
		return 0.08803927464621536*(temp(h)/216.65)**(-9.80665/(0.001*287.058)-1.0)
	elif (h > 32000.0 and h <= 47000.0):
		return 0.013226089459630561*(temp(h)/228.65)**(-9.80665/(0.0028*287.058)-1.0)
	elif (h > 47000.0 and h <= 51000.0):
		return 0.0014277005769293204*np.exp((-9.80665/(270.65*287.058))*(h - 47000.0))
	elif (h > 51000.0 and h <= 71000.0):
		return 0.00086171381927349503*(temp(h)/270.65)**(-9.80665/(-0.0028*287.058)-1.0) 
	elif (h > 71000.0 and h <= 84855.8345322628):
		return 6.422222128481488e-05 * (temp(h)/214.65)**(-9.80665/(-0.002*287.058)-1.0)
        elif (h > 84855.8345322628):
		return 6.9597971106303235e-06 * np.exp((-9.80665/(186.9467*287.058))*(h - 84852.0))

# Pressure for given altitude (def. inp. meters; def. out. Pascales))
def pres(h):
	try:
		if (float(h) < -610): 
			raise ValueError, 'Input out of bounds'
			return
	except TypeError:
		raise TypeError, 'Invalid input type'
		return
	if (h >= -610 and h <= 11000.0):
		return 101325*(temp(h)/288.15)**(-9.80665/(-0.0065*287.058))
	elif (h > 11000.0 and h <= 20000.0):
		return 22632.646369333983*np.exp((-9.80665/(216.65*287.058))*(h - 11000.0))
	elif (h > 20000.0 and h <= 32000.0):
		return 5475.162948547324*(temp(h)/216.65)**(-9.80665/(0.001*287.058))
	elif (h > 32000.0 and h <= 47000.0):
		return 868.0896162776735*(temp(h)/228.65)**(-9.80665/(0.0028*287.058))
	elif (h > 47000.0 and h <= 51000.0):
		return 110.91928623657715*np.exp((-9.80665/(270.65*287.058))*(h - 47000.0))
	elif (h > 51000.0 and h <= 71000.0):
		return 66.947288050821257*(temp(h)/270.65)**(-9.80665/(-0.0028*287.058)) 
	elif (h > 71000.0 and h <= 84855.8345322628):
		return 3.9571099295985768*(temp(h)/214.65)**(-9.80665/(-0.002*287.058))
	elif (h > 84855.8345322628):
		return 0.3734876816879029*np.exp((-9.80665/(186.9467*287.058))*(h - 84852.0))
