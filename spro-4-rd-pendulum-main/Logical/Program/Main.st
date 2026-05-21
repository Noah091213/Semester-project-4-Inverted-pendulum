
PROGRAM _INIT
	
	Q_vel_cart := 0;
	I_pos_pendulum;
	reset_pendulum_position;
	I_pos_cart;
	reset_cart_position;

	// PID K Values
	Kp_cart := -0.6000;
	Ki_cart := -0.2000;
	Kd_cart := -0.2000;

	Kp_pen := 76.63;
	Ki_pen := 0.00;
	Kd_pen := 11.59;
	
END_PROGRAM

PROGRAM _CYCLIC
	
	// LQR/LQI calculations
	x := (I_pos_cart - x_enc_middle) / x_enc_scale;
	x_dot := (x - x_old) / Ts;
			
	theta := (((I_pos_pendulum + pendulum_offset)  MOD 4096) - theta_enc_middle) * theta_enc_scale;
	theta_dot := (theta - theta_old) / Ts;
			
	filt_x_dot := (alpha * x_dot) + (( 1.0 - alpha) * filt_x_dot);
	filt_theta_dot := (alpha * theta_dot) + ((1.0 - alpha) * filt_theta_dot);
	
	// Swing calculations

	theta_offset := ((I_pos_pendulum + pendulum_offset - theta_enc_middle) MOD 4096) * theta_enc_scale;

	theta_cos := COS(theta_offset);
	
	
	IF theta_desturb THEN
		theta := theta + amPI/24;
	END_IF
	

	// Handle EStop
	IF(SI_estop_A AND SI_estop_B) THEN
		Q_drive_enable := TRUE;
	ELSE 
		Q_drive_enable  := FALSE;
		Q_reset_LED := FALSE;
		systemState := idle;
	END_IF
	
	// Handle Reset Btn
	IF(SI_reset_btn) THEN
		opc_step := 0;
		count := count + 1;
		IF (count = 10) THEN
			Q_drive_faultreset := NOT Q_drive_faultreset;
			count := 0;
		END_IF
	ELSIF I_drive_no_fault THEN
		Q_reset_LED := TRUE;
	ELSE
		led_ton(IN:=NOT led_ton.Q, PT := T#200ms);
		IF(led_ton.Q) THEN
			Q_reset_LED := NOT Q_reset_LED;
		END_IF
	END_IF

	CASE systemState OF // Main state machine
		idle:
			resetTimer.IN := FALSE;
			Q_vel_cart := 0;
			reset_pendulum_position := FALSE;
			reset_cart_position := FALSE;
			Q_drive_faultreset := NOT Q_drive_faultreset;
			
			count := 0;
			
			opc_state := 0;
			opc_step := 0;
			
			// Cascade reset
			error_x_old := 0;
			error_x := 0;
			u_i_cart := 0;
			u_d_cart := 0;
			u_d_old_cart := 0;
			u_cart := 0;
			
			error_theta := 0;
			error_theta_old := 0;
			u_i_pen := 0;
			u_d_pen := 0;
			u_d_old_pen := 0;
			
			// LQR reset
			theta_old := 0;
			x_old := 0;
			x_dot := 0;
			theta_dot := 0;
			filt_x_dot := 0;
			filt_theta_dot := 0;
			
		reset: // Position reset
			
			opc_state := 1;
			
			IF NOT I_endstop_positive THEN
				Q_vel_cart := 0;
			ELSE
				Q_vel_cart := 4000;
			END_IF
			
			resetTimer.IN := (Q_vel_cart = 0); // Timer counts as long as vel is 0
			resetTimer.PT := T#1s; // Timer set to 1 second
			resetTimer();
			
			IF resetTimer.Q THEN
				reset_cart_position := TRUE;
				
				systemState := center;
				
				resetTimer.IN := FALSE;
			END_IF
			
			Q_drive_faultreset := NOT Q_drive_faultreset;
			
		swingWait:
			
			opc_state := 5;
			
			Q_vel_cart := 0;
			
			IF (ABS(theta) < (amPI / 6)) AND (ABS(theta_dot) < 0.5) THEN
				systemState := controller;
			END_IF
			
			resetTimer.IN := (ABS(theta) > (amPI / 3)); // Timer counts as long as vel is 0
			resetTimer.PT := T#0.5s; // Timer set to 1 second
			resetTimer();
			
			IF resetTimer.Q THEN
				systemState := swing;
				
				resetTimer.IN := FALSE;
			END_IF
			
		swing:
			
			opc_state := 6;

			theta_dot := (theta_offset - theta_offset_old) / Ts;
			
			filt_theta_offset_dot := (alpha * theta_dot) + ((1.0 - alpha) * filt_theta_offset_dot);

			// 1. Calculate Total Energy (Kinetic + Potential)
			energy := (0.5 * I_p * filt_theta_offset_dot * filt_theta_offset_dot) + (m_p * g * l * (theta_cos - 1.0));

			// 2. Compute Energy Error (Target is 0.0 energy at the top)
			error_energy := 0.0 - energy;

			// 3. Determine push direction based on movement phase
			IF (filt_theta_offset_dot * theta_cos) > 0.0 THEN
				direction := 1.0;
			ELSIF (filt_theta_offset_dot * theta_cos) < 0.0 THEN
				direction := -1.0;
			ELSE
				direction := 0.0;
			END_IF
			
			// 4. Calculate Control Effort (2 * x is -kx)
			u := energy_gain * error_energy * direction + 1.5 * x;
			
			// --- CATCH CONDITION --
			// If angle is within ~30 degrees and velocity is low
			IF (ABS(theta) < (amPI / 6)) AND (ABS(theta_dot) < 1) THEN
				systemState := controller;
			END_IF
			
			// Cascade reset
			error_x_old := 0;
			error_x := 0;
			u_i_cart := 0;
			u_d_cart := 0;
			u_d_old_cart := 0;
			u_cart := 0;
			
			error_theta := 0;
			error_theta_old := 0;
			u_i_pen := 0;
			u_d_pen := 0;
			u_d_old_pen := 0;
			
		cascade:
			opc_state := 10;

			// --- CART (Outer Loop) ---
			error_x := -x;

			// Integral with Ts
			u_i_cart := u_i_cart + Ki_cart * Ts * ((error_x + error_x_old) / 2);

			// Derivative
			u_d_cart := (Kd_cart * (error_x - error_x_old)) / Ts;

			u_cart := Kp_cart * error_x + u_i_cart + u_d_cart;

			// Save state for next cycle
			error_x_old := error_x;
			u_d_old_cart := u_d_cart;


			// --- PENDULUM (Inner Loop) ---

			// The Cart output defines the required tilt angle
			error_theta := - u_cart - theta;

			// Derivative
			u_d_pen := (Kd_pen * (error_theta - error_theta_old)) / Ts;

			u := Kp_pen * error_theta + u_d_pen;

			// Save state for next cycle
			error_theta_old := error_theta;
			u_d_old_pen := u_d_pen;
			
			
		lqr:
			
			// LQR K Values
			K1 := 28.1045;
			K2 := 29.5079;
			K3 := 72.0365;
			K4 := 9.3493;
			
			opc_state := 20;
			
			u := - (( K1 * x ) + ( K2 * filt_x_dot ) + ( K3 * theta ) + ( K4 * filt_theta_dot ));
			
		lqi:
			
			// LQI K Values
			K1 :=  39.6;
			K2 :=  32.6;
			K3 :=  76.6;
			K4 :=  9.9;
			K5 := 14.0;
			
			opc_state := 30;
			
			error_x := error_x + (x * Ts);

			IF error_x > 5 THEN
				error_x := 5;
			ELSIF error_x < -5 THEN
				error_x := -5;
			END_IF;
			
			u := - (( K1 * x ) + ( K2 * filt_x_dot ) + ( K3 * theta ) + ( K4 * filt_theta_dot ) + (K5 * error_x));
		
		center:
			
			opc_state := 2;
			
			Q_vel_cart := -4000;
			IF (I_pos_cart > x_enc_middle) THEN
				systemState := swingWait;
			END_IF
			
			reset_pendulum_position := FALSE;
			reset_cart_position := FALSE;
			Q_drive_faultreset := NOT Q_drive_faultreset;
		
		pendulReset:
			
			opc_state := -1;
			
			reset_pendulum_position := TRUE;
			systemState := idle;
						
	END_CASE
	
	IF theta_desturb THEN
		opc_state := opc_state + 4;
		theta := theta - amPI/24;
		count := count + 1;
		
		IF count = 50 THEN
			theta_desturb := FALSE;
			count := 0;
		END_IF
		
	END_IF
	
	IF push THEN
		u := u + 6;
		count := count + 1;
		opc_state := opc_state + 1;
		
		IF count = 50 THEN
			opc_state := opc_state - 1;
			push := FALSE;
			count := 0;
		END_IF
		
	END_IF
	
	IF step THEN
		u := u + 8;
		count := count + 1;
		opc_state := opc_state + 2;
		
		IF count = 10000 THEN
			step := FALSE;
			count := 0;
		END_IF
		
	END_IF 

	IF (systemState = swing OR systemState = lqi OR systemState = lqr OR systemState = cascade) THEN
		Q_vel_cart := REAL_TO_INT(u * u_scale);
	END_IF
	
	IF (ABS(theta) > (amPI / 4) AND (systemState = lqi OR systemState = lqr OR systemState = cascade)) THEN
		systemState := swingWait;
	END_IF

	IF (Q_vel_cart > max_torque) THEN
		Q_vel_cart := max_torque;
	ELSIF (Q_vel_cart < -max_torque) THEN
		Q_vel_cart := -max_torque;
	END_IF
	
	IF ((NOT I_endstop_positive OR NOT I_endstop_negative) AND (systemState = swing OR systemState = lqi OR systemState = lqr OR systemState = cascade)) THEN
		Q_vel_cart := 0;
		systemState := idle;
	END_IF
	
	IF ((I_pos_cart > 360000 OR I_pos_cart < 20000) AND (systemState = swing OR systemState = lqi OR systemState = lqr OR systemState = cascade)) THEN
		Q_vel_cart := 0;
		systemState := idle;
	END_IF
	
	
	
	x_old := x;
	theta_old := theta;	
	theta_offset_old := theta_offset;
	
	opc_step := opc_step + 1;
	opc_x := x;
	opc_theta := theta;
	opc_u := u;
	
	 
END_PROGRAM

PROGRAM _EXIT

	
	 
END_PROGRAM

