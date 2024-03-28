from scipy.linalg import expm
from scipy.interpolate import make_interp_spline
import numpy as np
from pylinode import (
  linear_ivp_fundmat,
  solve_linear_ivp_normalized,
  solve_linear_ivp,
  linear_ivp_fundmat_normalized,
)
import mpmath as mp


def test_fundamental_solution_lti():
  T = 4.
  t = np.linspace(0, T, 100)
  np.random.seed(0)
  A = np.random.normal(size=(5,5))
  Afun = lambda t: A
  X = linear_ivp_fundmat(Afun, t, max_step=1e-3)
  F1 = expm(A * T)
  F2 = X[-1]
  assert np.allclose(F1, F2)

def test_fundamental_solution_ltv():
  T = 4.
  t = np.linspace(0, T, 100)
  dim = 6
  np.random.seed(0)
  S = np.random.normal(size=(dim, dim))
  Q = np.random.normal(size=(dim, dim))
  S = S - S.T

  def Afun(t):
    R = expm(S * t)
    dR = S @ expm(S * t)
    A = (dR + R @ Q) @ R.T
    return A

  X1 = linear_ivp_fundmat(Afun, t, max_step=1e-3)
  X2 = [expm(S * w) @ expm(Q * w) for w in t]
  assert np.allclose(X1, X2)

def test_solve_ltv():
  T = 4.
  t = np.linspace(0, T, 100)
  dim = 6
  np.random.seed(0)
  S = np.random.normal(size=(dim, dim))
  Q = np.random.normal(size=(dim, dim))
  S = S - S.T

  def Afun(t):
    R = expm(S * t)
    dR = S @ expm(S * t)
    A = (dR + R @ Q) @ R.T
    return A

  x0 = np.random.normal(size=(dim,))
  x1 = solve_linear_ivp(Afun, t, x0, max_step=1e-3)
  x2 = [expm(S * w) @ expm(Q * w) @ x0 for w in t]
  assert np.allclose(x1, x2)

def test_solve_ltv_normalized():
  T = 4.
  t = np.linspace(0, T, 100)
  dim = 5
  np.random.seed(0)
  S = np.random.normal(size=(dim, dim))
  Q = np.random.normal(size=(dim, dim))
  S = S - S.T

  def Afun(t):
    R = expm(S * t)
    dR = S @ expm(S * t)
    A = (dR + R @ Q) @ R.T
    return A

  x0 = np.random.normal(size=(dim,))
  y,l = solve_linear_ivp_normalized(Afun, t, x0, max_step=1e-3)
  x1 = y * np.exp(l[:,np.newaxis])
  x2 = [expm(S * w) @ expm(Q * w) @ x0 for w in t]
  assert np.allclose(x1, x2)


def test_fundamental_ltv_normalized():
  T = 5.
  t = np.linspace(0, T, 100)
  np.random.seed(0)
  Q = np.diag([9., -12., 2., -2., 4.])
  dim = Q.shape[0]
  S = np.random.normal(size=(dim, dim))
  S = S - S.T
  R0 = expm(S * 1.2)

  def Afun(t):
    R = R0 @ expm(S * t)
    dR = R0 @ S @ expm(S * t)
    A = (dR + R @ Q) @ R.T
    return A

  Y,l = linear_ivp_fundmat_normalized(Afun, t, max_step=1e-3)
  np.set_printoptions(suppress=True, linewidth=200)
  F = R0 @ expm(S * T) @ expm(Q * T)

  for i in range(dim):
    x1 = F[:,i]
    x2 = Y[-1,:,i] * np.exp(l[-1,i])
    np.allclose(x1, x2)

    x1 = F[:,i] / np.linalg.norm(F[:,i])
    x2 = Y[-1,:,i]
    np.allclose(x1, x2)

def test_monodromy_matrix_simple():
  T = 2 * np.pi
  t = np.linspace(0, T, 100)
  np.random.seed(0)
  Q = np.diag([9., -12., 2., -2., 4.])
  dim = Q.shape[0]
  S = np.random.normal(size=(dim, dim))
  S = S - S.T
  R0 = np.random.normal(size=(dim,dim))
  R0 = expm(S * 1.3)

  def Afun(t):
    R = R0 @ expm(S * t)
    dR = R0 @ S @ expm(S * t)
    A = (dR + R @ Q) @ R.T
    return A

  X = linear_ivp_fundmat(Afun, t, max_step=1e-3)
  monodromy_x = expm(Q * T)
  RT = R0 @ expm(S * T)
  monodromy_z = RT @ monodromy_x @ np.linalg.inv(R0)
  assert np.allclose(X[-1], monodromy_z)
