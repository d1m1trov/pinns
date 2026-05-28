"""
Physics-Informed Neural Network for the 1D Heat Equation
=========================================================
PDE  : u_t = alpha * u_xx,   x in [0, L],  t in [0, T]
IC   : u(0, x) = sin(pi*x/L)
BC   : u(t, 0) = 0,  u(t, L) = 0  (Dirichlet)
Exact: u(t, x) = exp(-alpha*(pi/L)^2 * t) * sin(pi*x/L)

Parameters used: alpha=1, L=1, T=1

Follows the continuous-time PINN framework of:
  Raissi, Perdikaris & Karniadakis (2019)
  "Physics-informed neural networks: A deep learning framework
   for solving forward and inverse problems involving nonlinear PDEs"
  Journal of Computational Physics, 378, 686-707.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.optimize import minimize
from pyDOE import lhs
import time

# ── TF2 compatibility: use the full TF1 API via tf.compat.v1 ─────────────────
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
tf1.reset_default_graph()

# Reproducibility
np.random.seed(1234)
tf1.set_random_seed(1234)


# ═════════════════════════════════════════════════════════════════════════════
class PhysicsInformedNN:
    """
    Continuous-time PINN for the 1D heat equation  u_t = alpha * u_xx.

    Network : (x, t)  -->  u(x, t)   [scalar output, tanh activations]

    Loss = MSE_0  (initial condition)
         + MSE_b  (Dirichlet boundary conditions)
         + MSE_f  (PDE residual at collocation points)
    """

    # ── Construction ─────────────────────────────────────────────────────────
    def __init__(self, X0, u0, X_lb, X_ub, X_f, layers, lb, ub, alpha=1.0):
        """
        Parameters
        ----------
        X0    : (N0, 2)  [x, 0]  initial-condition collocation points
        u0    : (N0, 1)  observed values u(0, x)
        X_lb  : (Nb, 2)  [0, t]  left-boundary points
        X_ub  : (Nb, 2)  [L, t]  right-boundary points
        X_f   : (Nf, 2)  [x, t]  interior collocation points
        layers: list of ints, e.g. [2, 100, 100, 100, 100, 1]
        lb    : (2,)  lower bound [x_min, t_min]
        ub    : (2,)  upper bound [x_max, t_max]
        alpha : thermal diffusivity
        """
        self.lb    = lb.astype(np.float32)
        self.ub    = ub.astype(np.float32)
        self.alpha = alpha

        # ── Training data ────────────────────────────────────────────────────
        self.x0   = X0[:, 0:1];  self.t0   = X0[:, 1:2];  self.u0 = u0
        self.x_lb = X_lb[:, 0:1]; self.t_lb = X_lb[:, 1:2]
        self.x_ub = X_ub[:, 0:1]; self.t_ub = X_ub[:, 1:2]
        self.x_f  = X_f[:, 0:1];  self.t_f  = X_f[:, 1:2]

        # ── Network parameters ───────────────────────────────────────────────
        self.layers = layers
        self.weights, self.biases = self._initialize_NN(layers)
        # Flat list of all trainable tf.Variables (used by L-BFGS-B wrapper)
        self._trainable = self.weights + self.biases

        # ── TF1 placeholders ─────────────────────────────────────────────────
        self.x0_tf   = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf   = tf1.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf   = tf1.placeholder(tf.float32, shape=[None, 1])
        self.x_lb_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.x_ub_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf1.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf  = tf1.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf  = tf1.placeholder(tf.float32, shape=[None, 1])

        # ── Computational graph ──────────────────────────────────────────────
        self.u0_pred   = self._net_u(self.x0_tf,   self.t0_tf)
        self.u_lb_pred = self._net_u(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred = self._net_u(self.x_ub_tf, self.t_ub_tf)
        self.f_pred    = self._net_f(self.x_f_tf,  self.t_f_tf)

        # ── Loss ─────────────────────────────────────────────────────────────
        self.loss = (
            tf1.reduce_mean(tf.square(self.u0_tf - self.u0_pred))   # MSE_0
          + tf1.reduce_mean(tf.square(self.u_lb_pred))               # MSE_b x=0
          + tf1.reduce_mean(tf.square(self.u_ub_pred))               # MSE_b x=L
          + tf1.reduce_mean(tf.square(self.f_pred))                  # MSE_f
        )

        # ── Gradients of loss w.r.t. all weights (needed for L-BFGS-B) ──────
        self._loss_grads = tf.gradients(self.loss, self._trainable)

        # ── Adam optimiser ───────────────────────────────────────────────────
        self._adam_op = tf1.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

        # ── Session ──────────────────────────────────────────────────────────
        self.sess = tf1.Session(
            config=tf1.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False)
        )
        self.sess.run(tf1.global_variables_initializer())

    # ── Private helpers ───────────────────────────────────────────────────────
    def _initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            W = self._xavier_init([layers[l], layers[l + 1]])
            b = tf1.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def _xavier_init(self, size):
        stddev = np.sqrt(2.0 / (size[0] + size[1]))
        return tf1.Variable(
            tf.random.truncated_normal(size, stddev=stddev, dtype=tf.float32)
        )

    def _neural_net(self, X):
        """Scaled forward pass: input mapped to [-1,1], tanh hidden layers."""
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    def _net_u(self, x, t):
        return self._neural_net(tf.concat([x, t], axis=1))

    def _net_f(self, x, t):
        """PDE residual  f = u_t - alpha * u_xx  via automatic differentiation."""
        u    = self._net_u(x, t)
        u_t  = tf.gradients(u, t)[0]
        u_x  = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        return u_t - self.alpha * u_xx

    def _feed_dict(self):
        return {
            self.x0_tf:   self.x0,   self.t0_tf:   self.t0,   self.u0_tf:   self.u0,
            self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
            self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
            self.x_f_tf:  self.x_f,  self.t_f_tf:  self.t_f,
        }

    # ── L-BFGS-B helpers ─────────────────────────────────────────────────────
    def _get_flat_weights(self):
        """Read all trainable variables into a single 1-D float64 array."""
        vals = self.sess.run(self._trainable)
        return np.concatenate([v.flatten() for v in vals]).astype(np.float64)

    def _set_flat_weights(self, flat_w):
        """Write a flat weight vector back into the TF variables."""
        idx = 0
        for var in self._trainable:
            shape = var.shape.as_list()
            size  = int(np.prod(shape))
            self.sess.run(var.assign(
                flat_w[idx: idx + size].reshape(shape).astype(np.float32)
            ))
            idx += size

    def _loss_and_flat_grad(self, flat_w):
        """
        Objective + gradient for scipy.optimize.minimize.
        scipy requires float64; TF uses float32 internally.
        """
        self._set_flat_weights(flat_w)
        fd = self._feed_dict()
        loss_val, grads = self.sess.run([self.loss, self._loss_grads], fd)
        flat_grad = np.concatenate([g.flatten() for g in grads]).astype(np.float64)
        return float(loss_val), flat_grad

    # ── Public interface ──────────────────────────────────────────────────────
    def train(self, n_adam=1000):
        """
        Faster low-accuracy training for demonstration purposes.
        """

        fd = self._feed_dict()

        # ── Phase 1: Adam ────────────────────────────────────────────────────
        print('── Phase 1: Adam (%d iterations) ──────────────────────' % n_adam)

        t0 = time.time()

        for it in range(n_adam):
            self.sess.run(self._adam_op, fd)

            if it % 200 == 0:
                loss_val = self.sess.run(self.loss, fd)
                print('  Adam  it %5d   loss = %.3e   (%.1f s)'
                      % (it, loss_val, time.time() - t0))
                t0 = time.time()

        # ── Phase 2: lightweight L-BFGS-B ───────────────────────────────────
        print('── Phase 2: L-BFGS-B ──────────────────────────────────')

        result = minimize(
            fun=self._loss_and_flat_grad,
            x0=self._get_flat_weights(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 200,
                'maxfun': 200,
                'ftol': 1e-6,
                'gtol': 1e-5
            }
        )

        self._set_flat_weights(result.x)

        print('  L-BFGS-B finished —', result.message)

    def predict(self, X_star):
        """
        Evaluate the network on arbitrary (x, t) points.

        Returns
        -------
        u_pred : (N, 1)  predicted solution
        f_pred : (N, 1)  PDE residual (should be ~0 inside the domain)
        """
        fd_u = {self.x0_tf:  X_star[:, 0:1], self.t0_tf:  X_star[:, 1:2]}
        fd_f = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}
        u_pred = self.sess.run(self.u0_pred, fd_u)
        f_pred = self.sess.run(self.f_pred,  fd_f)
        return u_pred, f_pred


# ═════════════════════════════════════════════════════════════════════════════
# Main script
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Problem parameters ────────────────────────────────────────────────────
    alpha = 1.0    # thermal diffusivity
    L     = 1.0    # spatial domain [0, L]
    T     = 1.0    # time domain    [0, T]

    lb = np.array([0.0, 0.0])
    ub = np.array([L,   T  ])

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    N0  = 40
    N_b = 40
    N_f = 2000

    # Smaller network for laptop training
    layers = [2, 20, 20, 20, 1]

    # ── Reference solution on a dense grid ───────────────────────────────────
    N_x, N_t = 256, 100
    x_vals = np.linspace(0, L, N_x)
    t_vals = np.linspace(0, T, N_t)
    X_grid, T_grid = np.meshgrid(x_vals, t_vals)          # (N_t, N_x)

    U_exact = (np.exp(-alpha * (np.pi / L)**2 * T_grid)
               * np.sin(np.pi * X_grid / L))

    X_star = np.hstack([X_grid.flatten()[:, None],
                        T_grid.flatten()[:, None]])
    u_star = U_exact.flatten()[:, None]

    # ── Initial condition data (t = 0) ────────────────────────────────────────
    idx_x = np.random.choice(N_x, N0, replace=False)
    x0    = x_vals[idx_x, None]
    u0    = np.sin(np.pi * x0 / L)
    X0    = np.hstack([x0, np.zeros_like(x0)])

    # ── Boundary condition data ───────────────────────────────────────────────
    idx_t = np.random.choice(N_t, N_b, replace=False)
    tb    = t_vals[idx_t, None]
    X_lb  = np.hstack([np.zeros_like(tb),      tb])   # x = 0
    X_ub  = np.hstack([L * np.ones_like(tb),   tb])   # x = L

    # ── Collocation points (Latin Hypercube) ──────────────────────────────────
    X_f = lb + (ub - lb) * lhs(2, N_f)

    # ── Build and train ───────────────────────────────────────────────────────
    model = PhysicsInformedNN(X0, u0, X_lb, X_ub, X_f, layers, lb, ub, alpha=alpha)

    print('\n══════════════════════════════════════════════════════════')
    print('  1D Heat Equation PINN')
    print('  alpha=%g  L=%g  T=%g' % (alpha, L, T))
    print('  N0=%d  N_b=%d  N_f=%d  layers=%s' % (N0, N_b, N_f, layers))
    print('══════════════════════════════════════════════════════════\n')

    t_start = time.time()
    model.train(n_adam=1000)
    print('\n  Total training time: %.2f s\n' % (time.time() - t_start))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('  Relative L2 error: %.4e' % error_u)

    U_pred = griddata(X_star, u_pred.flatten(), (X_grid, T_grid), method='cubic')

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))

    # Row 0: heatmaps ──────────────────────────────────────────────────────────
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.92, bottom=0.55, left=0.08, right=0.95, wspace=0.35)

    for col, (data, title) in enumerate([
            (U_exact, 'Exact $u(t,x)$'),
            (U_pred,  'PINN Prediction  (Rel. $L_2$: %.2e)' % error_u)]):
        ax = plt.subplot(gs0[0, col])
        h  = ax.imshow(data.T, interpolation='nearest', cmap='inferno',
                       extent=[0, T, 0, L], origin='lower', aspect='auto')
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
        ax.set_title(title, fontsize=10)
        if col == 1:
            ax.plot(X0[:, 1],  X0[:, 0],  'wx', ms=3, mew=0.8,
                    label='IC data (%d pts)' % N0, clip_on=False)
            ax.plot(X_lb[:, 1], X_lb[:, 0], 'w+', ms=3, mew=0.8,
                    label='BC data (%d pts)' % (2 * N_b), clip_on=False)
            ax.plot(X_ub[:, 1], X_ub[:, 0], 'w+', ms=3, mew=0.8, clip_on=False)
            for ts in [0.25, 0.50, 0.75]:
                ax.axvline(ts, color='w', ls='--', lw=0.9)
            ax.legend(loc='upper right', fontsize=7, frameon=False,
                      labelcolor='white')

    # Row 1: solution slices ───────────────────────────────────────────────────
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=0.44, bottom=0.08, left=0.08, right=0.95, wspace=0.45)

    for i, ts in enumerate([0.25, 0.50, 0.75]):
        t_idx = np.argmin(np.abs(t_vals - ts))
        ax = plt.subplot(gs1[0, i])
        ax.plot(x_vals, U_exact[t_idx, :], 'b-',  lw=2, label='Exact')
        ax.plot(x_vals, U_pred[t_idx, :],  'r--', lw=2, label='Prediction')
        ax.set_xlabel('$x$'); ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = %.2f$' % ts, fontsize=10)
        ax.set_xlim([0, L]); ax.set_ylim([-0.05, 1.05])
        ax.axis('square')
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                      ncol=2, frameon=False)

    fig.suptitle(
        r'PINN – 1D Heat Equation:  $u_t = \alpha\,u_{xx}$,'
        r'  $\alpha=%.1f$,  $u(0,x)=\sin(\pi x)$' % alpha,
        fontsize=11, y=0.98
    )

    plt.savefig('HeatEquation_PINN.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('HeatEquation_PINN.png', bbox_inches='tight', dpi=150)
    print('  Saved: HeatEquation_PINN.pdf / .png')
    plt.show()