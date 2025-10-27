import numpy as np 

from time import perf_counter

###########################################################
#DECLARING VARIABLES AND ACCESSING FILES###################
###########################################################


n = 102
nao = 102
nmo = 102
#N = n_orbitals**2


#coeff = np.fromfile('53.0') 
#print(coeff)
# 1-electron kinetic energy integrals
T = np.genfromtxt('t1e')
#print(T)
# 1-electron nuclear attraction integrals
V = np.genfromtxt('v1e')
# Two-electron integrals in chemist's notation (pq|rs)
print("Reading 2e file")
start_2e = perf_counter()
II = np.reshape(np.genfromtxt('int2e'),(n,n,n,n))
end_2e = perf_counter()
print(f"{start_2e-end_2e = :.3f}")


####################################################################################
#FORMING THE matrices with only the occupied orbital molecular orbital coefficients#
####################################################################################

C = np.fromfile('53.0')
CA = np.reshape(C[:nao*nmo], (nao, nmo), order='F')
CB = np.reshape(C[(nao*nmo):(2*nao*nmo)], (nao, nmo), order='F')

#coeffa = coeff[:n*n_orbitals]
#print(coeffa)
#matrixa = np.reshape(coeffa, (102,102), order ='F') 
#print(matrixa) 
#matrix A in the atomic orbital basis 


#coeffb = coeff[(n*n_orbitals):(2*n*n_orbitals)]
#print(coeffb) 
#matrixb = np.reshape (coeffb, (102,102), order = 'F')
#print(matrixb)

################
#overlap matrix# 
################


S = np.genfromtxt('ovlp')
over = np.reshape(S, (102,102), order="F")
# multiplying coefficients and overlap matrix to find if its 1
overlap = np.transpose(CA)@over@CA
print(overlap)
print(np.shape(overlap))
 
####################################################################
# GENERATE DENSITY MATRIX WITH OCCUPIED ORBITALS ONLY###############
####################################################################

#generate the alpha and beta density matrix (sum of alpha and beta density matrices) 
print("Generating Density Matrices")
D = CA@ np.transpose(CA) + CB@np.transpose(CB) 
Dalpha = CA@ np.transpose(CA) 
Dbeta = CB@ np.transpose(CB) 



###################################################################
#BUILDING COLOUMB AND EXCHANGE MATRICES############################
###################################################################


print("Einsum Couloumb/Exchange")
#coloumb matrix 
J = np.einsum('mnst, ts -> mn', II, D)
#exchange matrix
Ka = np.einsum('mtsn, ts -> mn', II, Dalpha)
Kb = np.einsum('mtsn, ts -> mn', II, Dbeta)
Fa = T + V + J - Ka

print(np.shape(Fa))

Fb = T + V + J - Kb



########################################################
# FC=SCe ###############################################
########################################################

try1 = Fa @ CA

try2 = over @ CA 

#print(try1) 

#print(try2) 

#print(np.divide(try1 , try2)) 



########################################################
#forming the fock matrix in the molecular orbital basis# 
########################################################

FMO = np.transpose(CA)@Fa@CA

print(FMO)


#qccoeff = np.fromfile('qccoeff')
#alpha_coeff, alpha_en, beta_coeff, beta_en = qccoeff[:N], qccoeff[N:N+n_orbitals], qccoeff[N+n_orbitals:2*N+n_orbitals], qccoeff[2*N+n_orbitals:]
#print(beta_en)
#alpha_coeff = np.reshape(alpha_coeff, (102,102), order="F")
#beta_coeff = np.reshape(beta_coeff, (102,102), order="F")
#alpha_en, beta_en = np.diag(alpha_en), np.diag(beta_en)

#ALPHA = over @ alpha_coeff @ alpha_en @ np.transpose(alpha_coeff) @ over
#BETA = over @ beta_coeff @ beta_en @ np.transpose(beta_coeff) @ over

