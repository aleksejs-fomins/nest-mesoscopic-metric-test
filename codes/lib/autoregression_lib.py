'''
Algorithm-AR(p)-Multivariate-WithTrials

x(t) = sum_i A(i)x(t-i) + B u(t)



'''

# Construct stacked past measurements
def AR_STACK_LAG(Y, p):
    nCh, nTr, nT = Y.shape()
    X = np.zeros((nCh*p, nTr, nT))
    for i in range(p):
        X[nCh*i:nCh*(i+1),:,i:] = Y[:,:,i:]
    return X

# Compute L2 norm of the fit
def AR_L2(Y, U, p, A, B):
    ABIG = np.hstack(A)
    XBIG = AR_STACK_LAG(Y, p)
    
    eps = Y - ABIG.dot(XBIG) - B.dot(U)
    return np.linalg.norm(eps)**2
    

def AR_MLE(Y, U, p):
    nCh, nTr, nT = Y.shape()
    
    # Construct stacked past measurements
    X = AR_STACK_LAG(Y, p)
    
    # Construct linear system for transition matrices
    A = np.einsum('ajk,bjk', X, Y)
    B = np.einsum('ajk,bjk', X, X)
    C = np.einsum('ajk,bjk', X, U)
    D = np.einsum('ajk,bjk', U, Y)
    E = C.T  #np.einsum('ajk,bjk', U, X)
    F = np.einsum('ajk,bjk', U, U)
    
    # Solve system
    BINV = np.linalg.inv(B)
    FINV = np.linalg.inv(F)
    TMP11 = A - C.dot(FINV.dot(D))
    TMP12 = B - C.dot(FINV.dot(E))
    TMP21 = D - E.dot(BINV.dot(A))
    TMP22 = F - E.dot(BINV.dot(C))

    REZ_A = np.linalg.inv(TMP12).dot(TMP11).T
    REZ_B = np.linalg.inv(TMP22).dot(TMP21).T
    
    # Unstack A matrices
    REZ_A_LST = [REZ_A[nCh*i:nCh*(i+1)] for i in range(p)]
    
    return REZ_A_LST, REZ_B