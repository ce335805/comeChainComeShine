import numpy as np
import itertools
import math
from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import eigsh 
import matplotlib.pyplot as plt 
from scipy.linalg import cosm


# OBC  = 0 
# PBC  = 1
def states_gen(L,N):
    which = np.array(list(itertools.combinations(range(L), N)))
    #print(which)
    grid = np.zeros((len(which), L), dtype="int8")

    # Magic
    grid[np.arange(len(which))[None].T, which] = 1
    
    return grid

def SzSz_L_red(BC, L, N, N_ph, OBC, PBC):
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N)*(N_ph+1))
    Id_M_L   = sparse.identity(dim)
    Mrx   = (dim, dim)
    init=sparse.csr_matrix(Mrx)
    for i in range(L-1):
        init += (c_dag_c_i_red(L, N, N_ph, i)-(0.5*Id_M_L)).dot(c_dag_c_i_red(L, N, N_ph, i+1)-(0.5*Id_M_L))
    init += (c_dag_c_i_red(L, N, N_ph, 0)-(0.5*Id_M_L)).dot(c_dag_c_i_red(L, N, N_ph, L-1)-(0.5*Id_M_L))

    return init


def Sz_i_red(BC, L, N, N_ph, OBC, PBC, i):
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N)*(N_ph+1))
    Id_M_L   = sparse.identity(dim)
    Mrx   = (dim, dim)
    init=sparse.csr_matrix(Mrx)
    
    init = (c_dag_c_i_red(L, N, N_ph, i)-(0.5*Id_M_L))
    
    return init


def trace_out(L, N, N_ph, OBC, PBC):
    Mrx_a_dag   = Mrces_a(N_ph)[0]
    Mrx_a       = Mrces_a(N_ph)[1]
    
    Mrx_a_dag_a = np.matmul(Mrx_a_dag, Mrx_a)
    
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N))
    Id_M_L   = sparse.identity(dim)
    
    a_dag_a = sparse.kron(Id_M_L, Mrx_a_dag_a)
    
    
def kin_L_red (BC, L, N, N_ph, g, OBC, PBC):
    
    states = states_gen(L , N) 
    #print(states)
    num_rows, num_cols = states.shape
    #print(states.shape)
    c_dag_c_kin_L = np.zeros((num_rows , num_rows))

    states_new = states.copy()
    #print(states_new)

    for i in range(num_rows): 
        #print('i is', i)
        for j in range(L - 1):
            #print('j is', j)
            if states_new[i][j] == 0 and states_new[i][j + 1] == 1:
                #print('old',states_new[i][:])
                states_new[i][j]     = 1
                states_new[i][j + 1] = 0
                #print('new', states_new[i][:])

                

                for k in range(num_rows):
                    
                    if np.array_equal(states_new[i][:],states[k][:]):
                        #print('that state is', states_new[i][:])
                        #print('coordinates are', i , k)
                        c_dag_c_kin_L[i][k] = 1
                        break
                
                states_new[i][:] = states[i][:]
                #print('new to old', states_new[i][:])
                #print('old', states[i][:])
        
        if BC == PBC:
            
            if states_new[i][L - 1] == 0 and states_new[i][0] == 1:
                
                states_new[i][0]         = 0  
                
                factor = np.sum(states_new[i][:])
                factor = (-1)**factor 
                
                states_new[i][L - 1]     = 1

                
                for k in range(num_rows):
                    if np.array_equal(states_new[i][:],states[k][:]):
                        #print('that state is', states_new[i][:])
                        #print('coordinates are', i , k)
                        c_dag_c_kin_L[i][k] = factor
                        break
                
                states_new[i][:] = states[i][:]
                #print('new to old', states_new[i][:])
                #print('old', states[i][:])
                
    
    Mrx_a_dag   = Mrces_a(N_ph)[0]
    Mrx_a       = Mrces_a(N_ph)[1]
    
    a_dag_plus_a = Mrx_a_dag + Mrx_a
    
    exp_iA = expm(1j * g * a_dag_plus_a)
    #exp_iA = cosm(g * a_dag_plus_a)
    
    c_dag_c_kin_L = sparse.kron(c_dag_c_kin_L, exp_iA)
    
    c_dag_c_kin_L += c_dag_c_kin_L.conjugate().T
    
    return c_dag_c_kin_L.tocsr()

def kin_L_red_rev (L, N, N_ph, g):
    
    states = states_gen(L , N) 
    #print(states)
    num_rows, num_cols = states.shape
    #print(states.shape)
    c_dag_c_kin_L = np.zeros((num_rows , num_rows))

    states_new = states.copy()
    #print(states_new)

    for i in range(num_rows): 
        #print('i is', i)
        for j in range(L - 1):
            #print('j is', j)
            if states_new[i][j] == 1 and states_new[i][j + 1] == 0:
                #print('old',states_new[i][:])
                states_new[i][j]     = 0
                states_new[i][j + 1] = 1
                #print('new', states_new[i][:])

                for k in range(num_rows):
                    if np.array_equal(states_new[i][:],states[k][:]):
                        #print('that state is', states_new[i][:])
                        #print('coordinates are', i , k)
                        c_dag_c_kin_L[i][k] = 1
                        break
                
                states_new[i][:] = states[i][:]
                #print('new to old', states_new[i][:])
                #print('old', states[i][:])

            else: 
                pass

    
    Mrx_a_dag   = Mrces_a(N_ph)[0]
    Mrx_a       = Mrces_a(N_ph)[1]
    
    a_dag_plus_a = Mrx_a_dag + Mrx_a
    
    exp_iA = expm(-1j * g * a_dag_plus_a)
    #exp_iA = cosm(g * a_dag_plus_a)
    
    c_dag_c_kin_L = sparse.kron(c_dag_c_kin_L, exp_iA)
    
    #c_dag_c_kin_L += c_dag_c_kin_L.conjugate().T
    
    return c_dag_c_kin_L.tocsr()

def WS_L_red (L, N, N_ph):
    
    states = states_gen(L , N) 
    #print(states)
    num_rows, num_cols = states.shape
    #print(states.shape)
    c_dag_c_WS_L = np.zeros((num_rows , num_rows))

    for i in range(num_rows): 
        #print('i is', i)
        cfnt = 0
        for j in range(L):
            #print('j is', j)
            if states[i][j] == 1 :
                cfnt  += (j + 1)
    
        c_dag_c_WS_L[i][i] = cfnt
        
    a_diag = np.eye(N_ph + 1)
    
    c_dag_c_WS_L = sparse.kron(c_dag_c_WS_L, a_diag)
            
    return c_dag_c_WS_L.tocsr()

def c_dag_c_i_red (L, N, N_ph, i):
    
    states = states_gen(L , N) 
    #print(states)
    num_rows, num_cols = states.shape
    #print(states.shape)
    c_dag_c_WS_L_i = np.zeros((num_rows , num_rows))

    for k in range(num_rows): 
        
        if states[k][i] == 1 :
            c_dag_c_WS_L_i[k][k] = 1
        
    a_diag = np.eye(N_ph + 1)
    
    c_dag_c_WS_L_i = sparse.kron(c_dag_c_WS_L_i, a_diag)
            
    return c_dag_c_WS_L_i.tocsr()




def Mrces_a(N_ph):
    a_dag_arr = np.zeros((N_ph + 1)**2)
    
    for i in range(N_ph):
        a_dag_arr[(i+1)* (N_ph + 1) + i] = np.sqrt(i + 1)

    Mrx_a_dag = np        . reshape   (a_dag_arr, (-1, N_ph + 1))
    Mrx_a     = Mrx_a_dag . transpose (                         )
        
    return Mrx_a_dag, Mrx_a





def a_dag_a_red (N_ph, L, N):
    
    Mrx_a_dag   = Mrces_a(N_ph)[0]
    Mrx_a       = Mrces_a(N_ph)[1]
    
    Mrx_a_dag_a = np.matmul(Mrx_a_dag, Mrx_a)
    
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N))
    Id_M_L   = sparse.identity(dim)
    
    a_dag_a = sparse.kron(Id_M_L, Mrx_a_dag_a)
    
    
    return a_dag_a.tocsr()

def a_dag_plus_a_red (N_ph, L, N):
    
    Mrx_a_dag   = Mrces_a(N_ph)[0]
    Mrx_a       = Mrces_a(N_ph)[1]
    
    a_dag_plus_a = Mrx_a_dag + Mrx_a
    dim =  np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N))
    Id_M_L   = sparse.identity(dim)
    
    a_dag_plus_a = sparse.kron(Id_M_L, a_dag_plus_a)
    
    return a_dag_plus_a.tocsr()


def n_phot_projector (N_ph, L, N, n):
    
    
    
    v= np.zeros(N_ph+1)
    v[n]=1
    v=np.reshape(v,(1,N_ph+1))
    
    prj=np.tensordot(v.T.conj(), v, axes=1)
    
    
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N))
    Id_M_L   = sparse.identity(dim)
    
    projector = sparse.kron(Id_M_L, prj)
    
    
    return projector.tocsr()


def expectation_value (op, A):
    
    sA     = sparse.csr_matrix(A)
    oc2    = op.dot(sA.T)
    #oc2    = oc2.dot(sA.T)
    oc2    = np.conj(sA).dot(oc2)
    value= np.real_if_close(oc2[0,0])
    return value

def expect_fast (op, A):
    
    val = np.real_if_close(np.tensordot(A.T.conj(),np.tensordot(op, A, 1), 1 )[0][0])
    return val



class Vector:
    def __init__(self, A):
        self.v=sparse.csr_matrix(A)
    def expectation_value(self, op):
         oc2    = op.dot(self.v.T)
         oc2    = np.conj(self.v).dot(oc2)
         value= np.real_if_close(oc2[0,0])
         return value
    def apply(self, op):
        self.v=op.dot(self.v.T).transpose()
    def expectation_corr(self, opi, opj):
        corr_op=opi.dot(opj)
        avg1=self.expectation_value(opi)
        avg2=self.expectation_value(opj)
        exp_corr=self.expectation_value(corr_op)
        val=exp_corr-(avg1*avg2)
        return val

def time_evol (sA, U, del_t):
    
    #sA     = sparse.csr_matrix(A)
    #print(sA.shape)

    #U = sparse.linalg.expm(- H * del_t)
    #print(U.shape)
    sA_new = U.dot(sA.T)
    sA_new = sA_new.transpose()
    #A_new = sparse.tensordot(U,sA,axes=(1,0))
    return sA_new


# DEfinition of correlation Function
def Sz_i_Sz_j_red(BC, L, N, N_ph, OBC, PBC, i, j):
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N)*(N_ph+1))
    Id_M_L   = sparse.identity(dim)
    Mrx   = (dim, dim)
    init=sparse.csr_matrix(Mrx)
    init = (c_dag_c_i_red(L, N, N_ph, i)-(0.5*Id_M_L)).dot(c_dag_c_i_red(L, N, N_ph, j)-(0.5*Id_M_L))
    

    return init



def Sz_red(BC, L, N, N_ph, OBC, PBC, i):
    dim      = np.int(math.factorial(L)/math.factorial(L-N)/math.factorial(N)*(N_ph+1))
    Id_M_L   = sparse.identity(dim)
    Mrx   = (dim, dim)
    init=sparse.csr_matrix(Mrx)
    init = c_dag_c_i_red(L, N, N_ph, i)-(0.5*Id_M_L)
    

    return init
    


