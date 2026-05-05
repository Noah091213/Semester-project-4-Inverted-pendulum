import sympy as sp
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# ==========================================
# 1. SYMPY PHYSICS DERIVATION
# ==========================================
m_p, m_c, g, l, b, c = sp.symbols('m_p m_c g l b c', real=True, positive=True)
F, t = sp.symbols('F t')
theta = sp.symbols('theta', cls=sp.Function)(t)
theta_dot = sp.diff(theta, t)
x_c = sp.symbols('x_c', cls=sp.Function)(t)
x_c_dot = sp.diff(x_c, t)

ppdotsquare = (x_c_dot-l*sp.cos(theta)*theta_dot)**2+(-l*sp.sin(theta)*theta_dot)**2+(0)**2 # Pendulum position dot and squared

T = 1/2 * x_c_dot**2 * m_c + 1/2 * ppdotsquare * m_p    # Kinetic energy
U = m_p*g*l*sp.cos(theta)                               # Potential energy
L = T-U

# d/dt(dL/dx_c_dot) - dL/dx_c = Q
xrhs = L.diff(x_c)
xlhs = L.diff(x_c.diff(t)).diff(t)
trhs = L.diff(theta)
tlhs = L.diff(theta.diff(t)).diff(t)

Eq1 = sp.Eq(xlhs - xrhs, F - b*x_c_dot)
Eq2 = sp.Eq(tlhs - trhs, -c*theta_dot)

# Linearization
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

# Laplace transform
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

solution = sp.solve([Eq1_laplace, Eq2_laplace], (X_s, Theta_s))

# ==========================================
# 2. TRANSFER FUNCTIONS & PARAMETERS
# ==========================================
params = {
    m_p: 0.084, 
    m_c: 0.5, 
    g: 9.82, 
    l: 0.35/2, 
    b: 5.0, 
    c: 0.0012
}

# Inner Plant: Angle (Gt_s)
Gt_s = sp.collect(sp.simplify(solution[Theta_s] / F_s), s)
num_Gt, den_Gt = sp.fraction(sp.simplify(Gt_s.subs(params)))
pendulum = ct.TransferFunction([float(c) for c in sp.Poly(num_Gt, s).all_coeffs()], 
                               [float(c) for c in sp.Poly(den_Gt, s).all_coeffs()])

# Outer Plant: Position (Gx_s)
Gx_s = sp.collect(sp.simplify(solution[X_s] / F_s), s)
num_Gx, den_Gx = sp.fraction(sp.simplify(Gx_s.subs(params)))
cart = ct.TransferFunction([float(c) for c in sp.Poly(num_Gx, s).all_coeffs()], 
                           [float(c) for c in sp.Poly(den_Gx, s).all_coeffs()])

# ==========================================
# 3. REUSABLE AUTO-TUNE FUNCTION
# ==========================================
def tune_with_root_locus(system,    # Transfer function
                         Mp_tgt,    # Overshoot
                         ts_tgt,    # Settling time
                         alpha_tgt, # settling band
                         tr_tgt,    # Rise time
                         plot_name  # filename.png 
                         ):
    print(f"\n--- Tuning: {plot_name} ---")
    
    log_term = np.log(Mp_tgt / 100.0) / -np.pi
    zeta_min = np.sqrt((log_term**2) / (1 + log_term**2))
    sigma_min = -np.log(alpha_tgt / 100.0) / ts_tgt
    wn_min = 1.8 / tr_tgt
    
    print(f"Goals -> Zeta >= {zeta_min:.3f} | Sigma <= -{sigma_min:.3f} | Omega_n >= {wn_min:.3f}")

    plt.figure()
    ct.root_locus(ct.minreal(system))
    plt.grid(True)
    plt.savefig(plot_name, dpi=1000)
    plt.close()

    rl_data = ct.root_locus_map(ct.minreal(system))
    
    required_poles = sum(1 for p in rl_data.loci[0])

    for i, gain_val in enumerate(rl_data.gains):
        poles_at_this_gain = rl_data.loci[i]
        passed_poles = 0
        
        for pole in poles_at_this_gain:
            sigma, wn = -np.real(pole), np.abs(pole)
            zeta = sigma / wn if wn != 0 else 1.0
            
            if sigma >= sigma_min and wn >= wn_min and zeta >= zeta_min:
                passed_poles += 1

        if passed_poles >= required_poles:
            print(f"SUCCESS! All specs met at Master Gain: {gain_val:.3f}")
            return gain_val
            
    print("Failed to meet all specs. Adjust goals or magnets.")
    return None

# ==========================================
# 4. EXECUTE CASCADE TUNING
# ==========================================

# --- STEP A: TUNE INNER LOOP (Fast) ---
magnet_1 = 8.0
magnet_2 = 10.0

Kd_in = 1.0
Kp_in = magnet_1 + magnet_2
Ki_in = magnet_1 * magnet_2

inner_ratio = ct.TransferFunction([Kd_in, Kp_in, Ki_in], [1, 0])

inner_gain = tune_with_root_locus(
    system=pendulum * inner_ratio,
    Mp_tgt=10.0,
    ts_tgt=2.0,
    alpha_tgt=2.0,
    tr_tgt=0.5,
    plot_name="root_locus_inner.png",
)

# --- STEP B: TUNE OUTER LOOP (Slow & Steady) ---
if inner_gain is not None:
    # Build the active inner system
    C_inner = ct.TransferFunction([Kd_in*inner_gain, Kp_in*inner_gain, Ki_in*inner_gain], [1, 0])
    F_over_ThetaRef = ct.feedback(C_inner, pendulum)
    
    outer_plant = F_over_ThetaRef * cart
    
    # Define Outer "Lean-to-Steer" Ratio 
    Kd_out_ratio = -1.0 
    Kp_out_ratio = -0.5
    
    # THE FIX: Add a real-world low-pass filter pole [0.01, 1] to the denominator. 
    # This represents (0.01s + 1), which stops the "Equal Order" crash!
    outer_ratio = ct.TransferFunction([Kd_out_ratio, Kp_out_ratio], [0.01, 1])
    
    # Tune the Outer Loop
    outer_gain = tune_with_root_locus(
        system=outer_plant * outer_ratio,
        Mp_tgt=50.0, 
        ts_tgt=10.0, 
        alpha_tgt=20.0, 
        tr_tgt=10.0, 
        plot_name="root_locus_outer.png",
    )
    
    if outer_gain is not None:
        print("\nFULL CASCADE SYSTEM COMPLETED!")
        print("--- Final Physical Inner Knobs (Pendulum) ---")
        print(f"Kp = {Kp_in * inner_gain:.5f}")
        print(f"Ki = {Ki_in * inner_gain:.5f}")
        print(f"Kd = {Kd_in * inner_gain:.5f}")
        print("--- Final Physical Outer Knobs (Cart) ---")
        print(f"Kp = {Kp_out_ratio * outer_gain:.5f}")
        print(f"Kd = {Kd_out_ratio * outer_gain:.5f}")