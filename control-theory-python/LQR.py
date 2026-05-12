import sympy as sp
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# System Setup & Euler-Lagrange Derivation
# ----
m_p, m_c, g, l, b, c = sp.symbols('m_p m_c g l b c', real=True, positive=True)
F, t = sp.symbols('F t')

theta = sp.symbols('theta', cls=sp.Function)(t)
theta_dot = sp.diff(theta, t)
theta_ddot = sp.diff(theta_dot, t)

x_c = sp.symbols('x_c', cls=sp.Function)(t)
x_c_dot = sp.diff(x_c, t)
x_c_ddot = sp.diff(x_c_dot, t)

# Kinematics & Energy
ppdotsquare = (x_c_dot - l*sp.cos(theta)*theta_dot)**2 + (-l*sp.sin(theta)*theta_dot)**2
T = 1/2 * x_c_dot**2 * m_c + 1/2 * ppdotsquare * m_p
U = m_p * g * l * sp.cos(theta)
L = T - U

# Euler-Lagrange Equations
Eq1 = sp.Eq(L.diff(x_c_dot).diff(t) - L.diff(x_c), F - b*x_c_dot)
Eq2 = sp.Eq(L.diff(theta_dot).diff(t) - L.diff(theta), -c*theta_dot)

# Linearization
trig_approx = {sp.sin(theta): theta, sp.cos(theta): 1}
small_squares = {theta**2: 0, theta_dot**2: 0}

Eq1_linear = sp.simplify(Eq1.subs(trig_approx).expand().subs(small_squares))
Eq2_linear = sp.simplify(Eq2.subs(trig_approx).expand().subs(small_squares))

# State-Space Formulation
# ----
# Solve for accelerations to get into standard form: x_ddot = ... , theta_ddot = ...
accels = sp.solve([Eq1_linear, Eq2_linear], (x_c_ddot, theta_ddot))

# Define State Vector: X = [x_c, x_c_dot, theta, theta_dot]^T
# dX/dt = [x_c_dot, x_c_ddot, theta_dot, theta_ddot]^T
f1 = x_c_dot
f2 = accels[x_c_ddot]
f3 = theta_dot
f4 = accels[theta_ddot]

state_equations = sp.Matrix([f1, f2, f3, f4])
states = sp.Matrix([x_c, x_c_dot, theta, theta_dot])

# Extract A and B matrices using Jacobians
A_sym = state_equations.jacobian(states)
B_sym = state_equations.diff(F)

# Substitute physical parameters
params = {
    m_p: 0.084,
    m_c: 0.5,
    g: 9.82,
    l: 0.35/2,
    b: 5.0,
    c: 0.0012
}

A_num = np.array(A_sym.subs(params)).astype(np.float64)
B_num = np.array(B_sym.subs(params)).astype(np.float64)
C_num = np.eye(4) # We assume we can measure or estimate all 4 states
D_num = np.zeros((4, 1))

sys_c = ct.StateSpace(A_num, B_num, C_num, D_num)

# Discretization for B&R PLC
# ----
# B&R Cycle Time
Ts = 0.001 
sys_d = sys_c.sample(Ts, method='zoh') # Zero-Order Hold

# LQR Tuning
# ----
# Q Matrix: Penalizes state errors [x_c, x_c_dot, theta, theta_dot]
# R Matrix: Penalizes control effort (actuator force)

# Higher Q means "fix this error faster", Higher R means "use less force"
Q = np.diag([
    10.0,    # Cart position
    1.0,     # Cart velocity
    100.0,   # Pendulum angle (Critical to keep upright)
    10.0     # Pendulum angular velocity
])

# Define your physical limit (e.g., max 5 Newtons)
max_force = 17.0 

R = np.array([[1.0 / (max_force**2)]])

# Compute Discrete LQR Gains
K, S, E = ct.dlqr(sys_d, Q, R)

print("--- B&R PLC LQR Configuration ---")
print(f"Task Cycle Time (Ts): {Ts} seconds")
print(f"K1 (Cart Pos):      {K[0][0]:.4f}")
print(f"K2 (Cart Vel):      {K[0][1]:.4f}")
print(f"K3 (Angle):         {K[0][2]:.4f}")
print(f"K4 (Angle Vel):     {K[0][3]:.4f}")

# Simulation

# 1. Setup Simulation Parameters
sim_time = 7.0                      # Simulate for n seconds
n_steps = int(sim_time / Ts)        # Number of discrete loops
Ad = sys_d.A                        # Discrete A matrix
Bd = sys_d.B.flatten()              # Discrete B matrix (flattened for easy math)

# 2. Initial Condition
# Start with the pendulum leaning slightly (0.1 rad / ~5.7 degrees), everything else 0
x_current = np.array([0.0, 0.0, 1.0, 0.0])

# Arrays to store data for plotting
log_x = np.zeros((n_steps, 4))
log_u = np.zeros(n_steps)
log_t = np.arange(0, sim_time, Ts)

# 3. Main Simulation Loop (Mirroring the PLC)
for i in range(n_steps):
    # Calculate raw control effort from LQR: u = -K * x
    u_raw = -np.dot(K[0], x_current) 
    
    # --- ADD SATURATION HERE ---
    u_current = np.clip(u_raw, -max_force, max_force)
    # ---------------------------
    
    # Log the current state and the SATURATED input
    log_x[i] = x_current
    log_u[i] = u_current
    
    # Update state using the limited force
    x_current = np.dot(Ad, x_current) + (Bd * u_current)

# 4. Plot the Results
plt.figure(figsize=(10, 8))

# Plot Cart Position
plt.subplot(3, 1, 1)
plt.plot(log_t, log_x[:, 0], label='Cart Position [m]')
plt.grid(True)
plt.legend()
plt.title('Linear LQR Response to Initial Angle Error')

# Plot Pendulum Angle
plt.subplot(3, 1, 2)
plt.plot(log_t, log_x[:, 2], 'r', label='Pendulum Angle [rad]')
plt.grid(True)
plt.legend()

# Plot Control Force
plt.subplot(3, 1, 3)
plt.plot(log_t, log_u, 'k', label='Control Force [N]')
plt.xlabel('Time [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("lqr_simulation.png", dpi=300)
print("Saved simulation plot to lqr_simulation.png")