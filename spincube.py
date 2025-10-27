import numpy as np 
import time
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyscf import __config__
from pyscf.tools import cubegen

# Load MO coefficients from binary file (assumes Fortran-order, float64)
C = np.reshape(np.genfromtxt('minh101'), (20, 20), order='F')  # Shape (6,6)# need to make the shape not hard coded


# define the no. of electrons, AOs and MOs

ne = 10
nao = 20  # needs to be number of spatial orbitals( exclude spin orbitals)


# construct density matrix 

P = C[:,:ne]@C[:,:ne].T.conj()
print(P)


# build the density matrices with respect to the x, y and z axis 


# total charge density

PI = P[:nao, :nao] + P[nao: , nao:]
#print(PI)

# x axis 

Px = P[nao:, :nao] + P[:nao, nao:]   
#print(Px)

# y axis
# Py
def mulComplex(z1,z2): 
    return z1*z2 

z1 = complex(1)
Py = (P[:nao, nao:] - P[nao:, :nao])
# Py = mulComplex(z1,z2)

#print(Py)

#Py =  i*(P[:ne, ne:] - P[ne:, :ne])
# total density from the z axis
Pz = P[:nao, :nao] - P[nao:, nao:]
# print(Pz)


# build the cube files

# Define molecule
mol = gto.Mole()
mol.atom = '''
 H  0.00000000000  0.00000000000  0.00000000000
 H  1.00000000000  1.73205080757  0.00000000000
 H  2.00000000000  0.00000000000  0.00000000000
 H  3.00000000000  1.73205080757  0.00000000000
 H  4.00000000000  0.00000000000  0.00000000000
 H  2.00000000000  3.46410161514  0.00000000000
 H -1.00000000000 -1.73205080757  0.00000000000
 H  1.00000000000 -1.73205080757  0.00000000000
 H  3.00000000000 -1.73205080757  0.00000000000
 H  5.00000000000 -1.73205080757  0.00000000000
    '''
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()

cubegen.density(mol, 'spinPy.cube', Py)
cubegen.density(mol, 'spinPz.cube', Pz)
cubegen.density(mol, 'spinPx.cube', Px)
