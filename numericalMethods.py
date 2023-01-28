import numpy as np


def bisectionMethod(f, a, b, tol=1e-3, maxIter=100):
    """
    Bisection method for finding a root of a function f in the interval [a,b].

    Args:
        f (function): function
        a (float): left endpoint of interval
        b (float): right endpoint of interval
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-3.
        maxIter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        float: root of f(x) = 0
    """
    if f(a)*f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    for i in range(maxIter):
        c = (a + b)/2
        if f(c) == 0 or (b - a)/2 < tol:
            return c
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    raise ValueError("Method failed after %d iterations." % maxIter)


def fixedPointIteration(f, x0, tol=1e-6, maxIter=100):
    """
    Fixed point iteration method for solving f(x) = x

    Args:
        f (function): function
        x0 (float): initial guess
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-6.
        maxIter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        float: root of f(x) = x
    """

    for i in range(maxIter):
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            return x1, i+1
        x0 = x1

    raise ValueError("Method failed after %d iterations." % maxIter)

# newtons method


def newtonsMethod(x0, f, df, tol=1e-6, maxIter=100):
    """
    Newton's method for finding a root of a function f.

    Args:
        x0 (float): initial guess
        f (function): function
        df (function): derivative of function
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-6.
        maxIter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        float: root of f(x) = 0
    """

    xn = x0
    for i in range(maxIter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn
        dfxn = df(xn)
        if dfxn == 0:
            return None
        xn = xn - fxn/dfxn


def false_position(f, a, b, tol=1e-6, maxiter=100):
    for i in range(maxiter):
        fa = f(a)
        fb = f(b)
        c = a - fa*(b-a)/(fb-fa)
        fc = f(c)
        if abs(fc) < tol:
            return c
        if fa*fc < 0:
            b = c
        else:
            a = c

# secant method


def secantMethod(x0, x1, f, tol=1e-6, maxIter=100):
    """
    secant method for finding a root of a function f.

    Args:
        x0 (float): initial guess
        x1 (float): next guess
        f (function): function
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-6.
        maxIter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        _type_: _description_
    """
    x = x0
    for i in range(maxIter):
        x = x - f(x)*(x-x1)/(f(x)-f(x1))
        if abs(f(x)) < tol:
            break
    return x

### Interpolation methods###

# Newton's Divided Difference


def dividedDifference(x, y):
    """
    Newton's divided difference algorithm.

    Args:
        x (np.array): array of ordinal values
        y (np.array): array of functional values

    Returns:
        np.array: coefficients of the Newton polynomial
    """
    n = len(x)
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n-j):
            a = coef[i+1][j-1]
            b = coef[i][j-1]
            c = x[i+j] - x[i]
            coef[i][j] = (a - b)/(c)
    return coef

# Neville's Interpolation


def nevilleInterpolation(x, y, z):
    """
    Neville's method for interpolating a function y(x) at a point z.

    Args:
        x (np.array): array of ordinal values
        y (np.array): array of functional values
        z (float): point at which to interpolate

    Returns:
        float: interpolated value
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length.")
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            coef[i, j] = ((z - x[i - j])*coef[i, j - 1] + (x[i] - z)
                          * coef[i - 1, j - 1])/(x[i] - x[i - j])
    return coef


def HermiteInterpolation(x, y, dy):
    """
    Hermitian interpolation.

    Args:
        x (np.array): array of ordinal values
        y (np.array): array of functional values
        dy (np.array): array of functional derivatives

    Returns:
        np.array: coefficients of the Hermite polynomial
    """
    n = len(x)
    X = np.repeat(x, 2)
    Y = np.repeat(y, 2)
    q = np.zeros((2*n, 2*n+1))

    for i in range(0, 2 * n, 2):
        idx = i//2
        q[i][0] = x[idx]
        q[i+1][0] = x[idx]
        q[i][1] = y[idx]
        q[i+1][1] = y[idx]

    for i in range(2, 2*n+1):
        for j in range(1+(i-2), 2*n):
            if i == 2 and j % 2 == 1:
                q[j][i] = dy[j//2]
            else:
                q[j][i] = (q[j][i-1] - q[j-1][i-1]) / \
                    (q[j][0] - q[(j-1)-(i-2)][0])
    return q

# Cubic Spline Interpolation


def CubicSpline(x, a):
    """
    Cubic spline interpolation.

    Args:
        x (np.array): ordinal values
        a (np.array): function values

    Returns:
        np.array: coefficients of the cubic spline polynomial
    """
    n = len(x)
    h = np.diff(x)

    alpha = np.zeros(n)

    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])

    l = np.zeros(n)
    u = np.zeros(n)
    z = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    l[0] = 1

    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
    l[n-1] = 1

    for j in range(n-2, -1, -1):
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])

    return np.array([a, b, c, d])

### Integration methods###

# composite Simpson's rule


def composite_simpson(f, a, b, n):
    """
    Composite Simpson's rule for numerical integration.

    Args:
        f (function): function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        n (int): number of subintervals must be even.

    Returns:
        float: integral of f from a to b
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])

# composite trapezoidal rule


def composite_trapezoidal(f, a, b, n):
    """
    Composite trapezoidal rule for numerical integration.

    Args:
        f (function): function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        n (int): number of subintervals

    Returns:
        float: integral of f from a to b
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h / 2 * np.sum(y[0:-1] + y[1:])

# Composite Midpoint Rule


def composite_midpoint(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h*np.sum(y[1:-1])

# Romberg integration


def Romberg(f, a, b, n):
    """
    Romberg integration for numerical integration.

    Args:
        f (function): function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        n (int): number of subintervals

    Returns:
        np.array: integral of f from a to b
    """

    R = np.zeros((n, n))
    R[0, 0] = composite_trapezoidal(f, a, b, 1)
    for i in range(1, n):
        R[i, 0] = composite_trapezoidal(f, a, b, 2**i)
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    return R

# Simpsons Adaptive Quadrature


def AdaptiveQuadrature(f, a, b, tol=1e-6):
    """
    Adaptive quadrature for numerical integration.

    Args:
        f (function): function to integrate
        a (float): lower bound of integration
        b (float): upper bound of integration
        tol (_type_, optional): tolerance for stopping criterion. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    # Simpson's Rule
    def Simpson(f, a, b):
        h = (b - a) / 2
        return h / 3 * (f(a) + 4 * f(a + h) + f(b))

    # Recursive function
    def Recurse(f, a, b, tol):
        c = (a + b) / 2
        left = Simpson(f, a, c)
        right = Simpson(f, c, b)
        if abs(left + right - Simpson(f, a, b)) < 15 * tol:
            return left + right + (left + right - Simpson(f, a, b)) / 15
        return Recurse(f, a, c, tol / 2) + Recurse(f, c, b, tol / 2)

    return Recurse(f, a, b, tol)

# Gaussian Elimination


def GaussianElimination(A: np.array):
    """
    Guassian elimination for solving linear systems.

    Args:
        A (np.array): augmented matrix to be solved

    Raises:
        ValueError: matrix is singular

    Returns:
        np.array: solution vector
    """
    A = A.astype(float)
    m, n = A.shape
    n -= 1
    x = np.zeros(A.shape[0])

    for i in range(n):
        p = np.argmin(np.abs(np.ma.masked_where(A[i:, i] == 0, A[i:, i]))) + i
        if A[p, i] == 0:
            raise ValueError('Matrix is not unique')
        if p != i:
            A[[i, p]] = A[[p, i]]
        for j in range(i+1, n):
            A[j] = A[j] - A[i] * (A[j, i] / A[i, i])
    if A[n-1, n-1] == 0:
        raise ValueError('Matrix is not unique')
    x[n-1] = A[n-1, n] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (A[i, n] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    return x


# Gaussian Elimination with Partial Pivoting
def PartialPivoting(A: np.array):
    """
    Guassian elimination with partial pivoting for solving linear systems.

    Args:
        A (np.array): augmented matrix to be solved

    Raises:
        ValueError: matrix is singular

    Returns:
        np.array: solution vector
    """
    A = A.astype(float)
    m, n = A.shape
    n -= 1
    x = np.zeros(A.shape[0])

    for i in range(n):
        p = np.argmax(np.abs(A[i:, i])) + i
        if A[p, i] == 0:
            raise ValueError('Matrix is not unique')
        if p != i:
            A[[i, p]] = A[[p, i]]
        for j in range(i+1, n):
            A[j] = A[j] - A[i] * (A[j, i] / A[i, i])
    if A[n-1, n-1] == 0:
        raise ValueError('Matrix is not unique')
    x[n-1] = A[n-1, n] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (A[i, n] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    return x

# Gaussian Elimination with Scaled Partial Pivoting


def ScaledPartialPivoting(A: np.array):
    """
    Guassian elimination with scaled partial pivoting for solving linear systems.

    Args:
        A (np.array): augmented matrix to be solved

    Raises:
        ValueError: matrix is singular

    Returns:
        np.array: solution vector
    """
    A = A.astype(float)
    m, n = A.shape
    n -= 1
    x = np.zeros(A.shape[0])

    for i in range(n-1):
        p = np.argmax(np.abs(A[i:, i]) / np.max(A[i:, i])) + i
        if A[p, i] == 0:
            raise ValueError('Matrix is not unique')
        if p != i:
            A[[i, p]] = A[[p, i]]
        for j in range(i+1, n):
            A[j] = A[j] - A[i] * A[j, i] / A[i, i]
    if A[n-1, n-1] == 0:
        raise ValueError('Matrix is not unique')
    x[n-1] = A[n-1, n] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (A[i, n] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    return x

# LU decomposition


def LUdecomposition(A):
    """
    LU decomposition for solving linear systems.

    Args:
        A (np.array): nxn matrix

    Returns:
        np.array: L matrix
        np.array: U matrix
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i, n):
            if i == j:
                L[i, i] = 1
            else:
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i]))/U[i, i]
    return L, U


# LDL decomposition
def LDLdecomposition(A):
    """
    Compute the LDL decomposition of a square matrix.

    Args:
        A (np.array): array to be decomposed

    Raises:
        ValueError: matrix must be square

    Returns:
        list: list containing L, D, and L^T
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")
    l = np.eye(n)
    d = np.zeros((n, n))
    v = np.zeros(n)
    for i in range(n):
        for j in range(i):
            v[j] = l[i, j]*d[j, j]
        d[i, i] = A[i, i] - np.dot(l[i, :i], v[:i])
        for j in range(i+1, n):
            l[j, i] = (A[j, i] - np.dot(l[j, :i], v[:i]))/d[i, i]
    return l, d, np.transpose(l)

# Cholesky decomposition


def CholeskyDecomposition(A):
    """
    Compute the Cholesky decomposition of a square matrix.

    Args:
        A (np.array): array to be decomposed

    Returns:
        list: list containing L and L^T
    """
    n, m = A.shape
    if n != m:
        return
    l = np.zeros((n, n))
    l[0, 0] = np.sqrt(A[0, 0])

    for i in range(1, n):
        l[i, 0] = A[i, 0]/l[0, 0]

    for i in range(1, n-1):
        l[i, i] = np.sqrt(A[i, i] - np.dot(l[i, :i], l[i, :i]))
        for j in range(i+1, n):
            l[j, i] = (A[j, i] - np.dot(l[j, :i], l[i, :i]))/l[i, i]
    l[n-1, n-1] = np.sqrt(A[n-1, n-1] - np.dot(l[n-1, :n-1], l[n-1, :n-1]))
    return l, np.transpose(l)


def isDiagonallyDominant(A):
    """
    checks if array is strictly diagonally dominant

    Args:
        A (np.array): array to be checked

    Returns:
        bool: True if array is strictly diagonally dominant, False otherwise
    """
    for i in range(A.shape[0]):
        if abs(A[i, i]) < np.sum(abs(A[i, :])) - abs(A[i, i]):
            return False
    return True


def spectral_radius(A: np.array):
    """
    computes the spectral radius of a matrix

    Args:
        A (np.array): matrix to be checked

    Raises:
        ValueError: matrix must be square

    Returns:
        value: spectral radius of matrix
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    return np.max(np.abs(np.linalg.eigvals(A)))


def JacobiIteration(A: np.array, b: np.array, tol: float = 1e-10, maxIter: int = 1000):
    """
    computes the solution to a linear system using the Jacobi iteration method

    Args:
        A (np.array): A matrix
        b (np.array): b vector
        tol (float, optional): tolerance. Defaults to 1e-10.
        maxIter (int, optional): max number of iterations. Defaults to 1000.

    Returns:
        np.array: solution vector
    """
    A = np.array(A, dtype=np.float64)
    x = np.zeros_like(b, dtype=np.float64)  # initial guess
    T = A - np.diag(np.diag(A))  # T = A - D
    for k in range(maxIter):
        x_old = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diag(A)
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x
    return x


def GaussSeidelIteration(A: np.array, b: np.array, tol: float = 1e-10, maxIter: int = 1000):
    """
    Computes the solution to a linear system using the Gauss-Seidel iteration method

    Args:
        A (np.array): A matrix
        b (np.array): b vector
        tol (float, optional): tolerance. Defaults to 1e-10.
        maxIter (int, optional): max number of iterations. Defaults to 1000.

    Returns:
        np.array: solution vector
    """
    A = np.array(A, dtype=np.float64)
    x = np.zeros_like(b, dtype=np.float64)  # initial guess
    i = 0
    for k in range(maxIter):
        i += 1
        x_old = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) -
                    np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x
    return x


def inf_norm(A):
    column_sums = [sum(abs(A[:, i])) for i in range(A.shape[1])]
    return max(column_sums)
