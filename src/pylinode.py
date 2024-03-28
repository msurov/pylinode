import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple


def solve_linear_ivp(
    Afun : Callable[[float], np.ndarray],
    t_eval : np.ndarray,
    x0 : np.ndarray,
    **kwargs
  ) ->  np.ndarray:
  R"""
    Solve the IVP
    \[
      \dot{x} = A(t) x, \quad x(t_0) = x_0
    \]
  """
  n, = np.shape(x0)

  def rhs(t, x):
    dx = Afun(t) @ x
    return dx
  
  sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval, **kwargs)
  x = sol.y.T
  return x

def linear_ivp_fundmat(
    Afun : Callable[[float], np.ndarray],
    t_eval : np.ndarray,
    **kwargs
  ) -> np.ndarray:
  R"""
    Solve the matrix IVP
    \[
      \dot{X} = A(t) X, \quad X(t_0) = I
    \]
  """
  A0 = Afun(t_eval[0])
  n,_ = A0.shape

  def rhs(t, x):
    X = np.reshape(x, (n, n))
    dX = Afun(t) @ X
    dx = np.reshape(dX, (n**2,))
    return dx
  
  X0 = np.eye(n)
  x0 = np.reshape(X0, (n**2,))
  sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval, **kwargs)
  x = sol.y.T
  X = np.reshape(x, (-1, n, n))
  return X

def solve_linear_ivp_normalized(
    Afun : Callable[[float], np.ndarray], 
    t_eval : np.ndarray, 
    x0 : np.ndarray, 
    **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
  R"""
    Find normalized solution vector  
    \[
      y(t) = \frac{ x(t) }{ \left\Vert x(t) \right\Vert }
    \]
    and log of its norm
    \[
      l(t)=\log\left\Vert x(t)\right\Vert 
    \]
  """
  n, = np.shape(x0)
  I = np.eye(n)

  def rhs(t, state):
    y = state[:n]
    A = Afun(t)
    dy = (A - I * (y.T @ (A.T + A) @ y) / 2) @ y
    dl = y.T @ (A.T + A) @ y / 2
    dstate = np.zeros(n + 1)
    dstate[:n] = dy
    dstate[n] = dl
    return dstate

  y0 = np.zeros(n + 1)
  y0[:n] = x0
  y0[n] = 0
  sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, **kwargs)
  x = sol.y[0:n].T
  l = sol.y[n]
  return x, l

def linear_ivp_fundmat_normalized(
    Afun : Callable[[float], np.ndarray],
    t_eval : np.ndarray,
    **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
  R"""
    Find normalized solution vectors
    \[
      y(t) = \frac{ x(t) }{ \left\Vert x(t) \right\Vert }
    \]
    and log of its norm
    \[
      l(t)=\log\left\Vert x(t)\right\Vert 
    \]
  """
  n,_ = np.shape(Afun(t_eval[0]))
  I = np.eye(n)

  def rhs(t, state):
    dstate = np.zeros(state.shape)
    A = Afun(t)
    for j in range(n):
      y = state[j * n : (j + 1) * n]
      dl = y.T @ (A.T + A) @ y / 2
      dy = (A - I * dl) @ y
      dstate[j * n : (j + 1) * n] = dy
      dstate[n**2 + j] = dl
    return dstate

  state0 = np.zeros(n**2 + n)
  state0[0:n**2] = np.eye(n).reshape(n**2)
  state0[n**2:] = 0
  sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], state0, t_eval=t_eval, **kwargs)
  Y = sol.y[0:n**2,:].reshape((n, n, -1)).transpose([2,1,0])
  l = sol.y[n**2:].T
  return Y,l
