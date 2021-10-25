import sys

from functions_ED import Vector, states_gen, expectation_value, c_dag_c_i_red, SzSz_L_red, kin_L_red, a_dag_a_red, a_dag_plus_a_red, Sz_i_Sz_j_red, Sz_red, Sz_i_red, n_phot_projector

import scipy
from scipy.sparse.linalg import eigsh 
import numpy as np
from scipy.sparse.linalg import expm
from scipy.linalg import expm as expM
from scipy.linalg import sinm, cosm
from time import time
import matplotlib.pyplot as plt
import math

L=10
N=int(L/2)
N_ph=20
g=1

J=1


Omega=1
PBC=1
OBC=0
BC=PBC


def H(J, g, Omega):
   H=-(J)*kin_L_red(BC, L, N, N_ph, g/np.sqrt(L), OBC, PBC)+Omega*a_dag_a_red(N_ph, L, N)
   return H

fig, ax = plt.subplots(dpi = 500)


H0=H(J, g, Omega)

w, v= eigsh(H0, 1, which='SA')
gs=Vector(v[:, 0])
nps = np.arange(0, 21)

phots=[]
for i in range(N_ph+1):
    phots.append(float(gs.expectation_value(n_phot_projector(N_ph, L, N, i))))
    

ax.plot(nps, phots, linestyle='',marker='x')
ax.set_yscale('log')

    



    
    
    
    
    








    

