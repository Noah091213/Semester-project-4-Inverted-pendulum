import sympy as sp
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Constants and calculations for energy (From blackboard)
m_p, m_c, g, l, b, c = sp.symbols('m_p m_c g l b c', real=True, positive=True)
F = sp.symbols('F')
t = sp.symbols('t')
theta = sp.symbols('theta', cls=sp.Function)
theta = theta(t)
theta_dot = sp.diff(theta, t)
x_c = sp.symbols('x_c', cls=sp.Function)
x_c = x_c(t)
x_c_dot = sp.diff(x_c, t)
ppdotsquare = (x_c_dot-l*sp.cos(theta)*theta_dot)**2+(-l*sp.sin(theta)*theta_dot)**2+(0)**2

# Euler-Lagrange
T = 1/2 * x_c_dot**2 * m_c + 1/2 * ppdotsquare * m_p    # Kinetic energy
U = m_p*g*l*sp.cos(theta)                               # Potential energy
L = T-U

xrhs = L.diff(x_c)
xlhs = L.diff(x_c.diff(t)).diff(t)
trhs = L.diff(theta)
tlhs = L.diff(theta.diff(t)).diff(t)

Eq1 = sp.Eq(xlhs - xrhs, F - b*x_c_dot)
Eq2 = sp.Eq(tlhs - trhs, -c*theta_dot)

# Linearization of E-L equations
trig_approx = {
    sp.sin(theta): theta,
    sp.cos(theta): 1
}

small_squares = {
    theta**2: 0,
    theta_dot**2: 0
}

Eq1_linear = sp.simplify(Eq1.subs(trig_approx).expand().subs(small_squares))
Eq2_linear = sp.simplify(Eq2.subs(trig_approx).expand().subs(small_squares))

# Laplace transform E-L equations to frequancy domain
s, X_s, Theta_s, F_s = sp.symbols('s X_s Theta_s F_s')

laplace_rules = {
    x_c.diff(t, 2): s**2 * X_s,
    x_c.diff(t): s * X_s,
    x_c: X_s,
    theta.diff(t, 2): s**2 * Theta_s,
    theta.diff(t): s * Theta_s,
    theta: Theta_s,
    F: F_s
}

Eq1_laplace = Eq1_linear.subs(laplace_rules)
Eq2_laplace = Eq2_linear.subs(laplace_rules)

solFD = sp.solve([Eq1_laplace, Eq2_laplace], (X_s, Theta_s))

# Transfer function
G_s = solFD[Theta_s] / F_s

G_s = sp.simplify(G_s)
G_s = sp.collect(G_s, s)

params = {
    m_p: 0.084,
    m_c: 0.5,
    g: 9.82,
    l: 0.35/2,
    b: 5.0,
    c: 0.0012
}

G_s_params = G_s.subs(params)
num_G, den_G = sp.fraction(sp.simplify(G_s_params))

num_G_ct = [float(c) for c in sp.Poly(num_G, s).all_coeffs()]
den_G_ct = [float(c) for c in sp.Poly(den_G, s).all_coeffs()]

pendulum = ct.TransferFunction(num_G_ct, den_G_ct)

# Calculate PID values for control transfer function using set 'magnets'/zeros
magnet_1 = 8.0
magnet_2 = 10.0

Kd = 1.0
Kp = magnet_1 + magnet_2
Ki = magnet_1 * magnet_2

print(f"\nUsing PID Values: Kd = {Kd}, Kp = {Kp}, Ki = {Ki}")

num_K = [Kd, Kp, Ki]
den_K = [1, 0]
control = ct.TransferFunction(num_K, den_K)

system = ct.minreal(pendulum * control)

# Set performance goals
Mp_target = 10.0    # Max 10% Overshoot
ts_target = 2.0    # Settle in 2.0 seconds
alpha = 2.0        # 2% settling band
tr_target = 0.5    # Rise time under 0.5 seconds

log_term = np.log(Mp_target / 100.0) / -np.pi
zeta_min = np.sqrt((log_term**2) / (1 + log_term**2))

sigma_min = -np.log(alpha / 100.0) / ts_target
wn_min = 1.8 / tr_target

print(f"\nTo meet goals, poles must have:")
print(f"Zeta (Damping) >= {zeta_min:.3f}")
print(f"Sigma (Real Part) <= -{sigma_min:.3f}")
print(f"Omega_n (Distance from center) >= {wn_min:.3f}")

# Tune PID values
print("\nScanning through root locus map...")
rl_data = ct.root_locus_map(system)
loci = rl_data.loci
gains = rl_data.gains

ct.root_locus(system)
plt.grid(True)
plt.savefig("root_locus.png", dpi=1000)
print("Computed and saved root_locus.png")

found_perfect_gain = False

for i, gain_val in enumerate(gains):
    poles_at_this_gain = loci[i]
    
    physical_poles_passed = 0
    total_physical_poles = 0
    
    for pole in poles_at_this_gain:
        total_physical_poles += 1
        
        # Calculate the traits of this specific pole
        sigma = -np.real(pole)                     # Distance to the left
        wn = np.abs(pole)                          # Distance from origin
        zeta = sigma / wn if wn != 0 else 1.0      # Angle ratio
        
        # Settling Time (Is it far enough left?)
        pass_ts = sigma >= sigma_min
        
        # Rise Time (Is it fast enough overall?)
        pass_tr = wn >= wn_min
        
        # Overshoot (Is the angle safe?)
        pass_Mp = zeta >= zeta_min
        
        if pass_ts and pass_tr and pass_Mp:
            physical_poles_passed += 1

    if physical_poles_passed >= 3:
        print(f"\nGain with poles passing the performance checks: {gain_val:.3f}")
        print("New PID values:")
        print(f"Kp = {Kp * gain_val:.3f}")
        print(f"Ki = {Ki * gain_val:.3f}")
        print(f"Kd = {Kd * gain_val:.3f}")
        
        found_perfect_gain = True
        break

if not found_perfect_gain:
    print("\nSystem could not meet all performance specs.")
    print("Try relaxing the goals (fx allow more overshoot or settling time),")
    print("or try moving the magnets (zeros) to different numbers!")