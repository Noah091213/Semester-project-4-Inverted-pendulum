
PROGRAM _INIT // Alt her bruges, må gerne få refactor men ikke ændre funktionalitet

	Q_vel_cart := 0;
	length := 0;
	I_pos_pendulum;
	reset_pendulum_position;
	I_pos_cart;
	reset_cart_position;
	systemState := idle;
	speed := -2000;
	
	scale := 1310;
	
END_PROGRAM

PROGRAM _CYCLIC

	// Handle EStop
	IF(SI_estop_A AND SI_estop_B) THEN
		Q_drive_enable := TRUE;
	ELSE 
		Q_drive_enable  := FALSE;
		Q_reset_LED := FALSE;
	END_IF
	
	// Handle Reset Btn
	IF(SI_reset_btn) THEN
		Q_drive_faultreset := NOT Q_drive_faultreset;
	ELSIF I_drive_no_fault THEN
		Q_reset_LED := TRUE;
	ELSE
		led_ton(IN:=NOT led_ton.Q, PT := T#200ms);
		IF(led_ton.Q) THEN
			Q_reset_LED := NOT Q_reset_LED;
		END_IF
	END_IF

	CASE systemState OF // Main state machine
		idle: // Bruges
			resetTimer.IN := FALSE;
			counter := 0;
			Q_vel_cart := 0;
			reset_pendulum_position := FALSE;
			reset_cart_position := FALSE;
			Q_drive_faultreset := NOT Q_drive_faultreset;
			
			e_prev := 0;
			Fd_prev := 0;
			e_sum := 0;
			
		reset: // Bruges
			IF NOT I_endstop_positive THEN
				Q_vel_cart := 0;
			ELSE
				Q_vel_cart := 4000;
			END_IF
			
			resetTimer.IN := (Q_vel_cart = 0); // Timer counts as long as vel is 0
			resetTimer.PT := T#1s; // Timer set to 1 second
			resetTimer();
			
			IF resetTimer.Q THEN
				reset_pendulum_position := TRUE;
				reset_cart_position := TRUE;
				
				systemState := idle;
			END_IF
			
			Q_drive_faultreset := NOT Q_drive_faultreset;
			
		manual: // Bruges ikke, var tænkt til kontrol med piletaster el.lign.
			IF NOT I_endstop_negative THEN
				systemState := idle;
			END_IF
			Q_vel_cart := -4000;
			Q_drive_faultreset := NOT Q_drive_faultreset;
			
		swing: // Bruges ikke, tænkt til "swing-up" sequence.
			Q_vel_cart := speed;
			counter := counter + 1;
			IF (counter = 9) THEN
				pos_start := I_pos_cart;
				resetTimer.IN := TRUE;
				resetTimer.PT := T#5s;
			END_IF
			
			resetTimer();
			
			IF resetTimer.Q THEN
				pos_end := I_pos_cart;
				systemState := idle;
			END_IF
			
			IF NOT I_endstop_negative THEN
				systemState := idle;
			END_IF
			
		controller: // Tidlig test controller, bør gemmes
			IF (I_pos_pendulum > 1700 AND I_pos_pendulum < 2350) THEN
				Q_vel_cart := speed * (DINT_TO_INT(I_pos_pendulum) - 2048);
			ELSE
				Q_vel_cart := 0;
			END_IF
			IF (Q_vel_cart > 10000) THEN
				Q_vel_cart := 10000;
			ELSIF (Q_vel_cart < -10000) THEN
				Q_vel_cart := -10000;
			END_IF
			IF NOT I_endstop_positive OR NOT I_endstop_negative THEN
				Q_vel_cart := 0;
			END_IF
			
		PDcontroller: // MAIN PID CONTROLLER - NO TOUCH!
			// Cart (outer loop)
			Kp_cart := -0.00234 * 50;
			Ki_cart := 0;
			Kd_cart := -0.00468 * 50;
			Tf_cart := 0.01;
			Ts_cart := 0.002;
			
			pos_ref := 190000;
			pos := I_pos_cart;
			
			e_cart := (pos_ref - pos) / 243500;
			
			e_sum_cart := e_sum_cart + (e_cart * Ts_cart);
			
			IF e_sum_cart > 100.0 THEN 
				e_sum_cart := 100.0; 
			ELSIF e_sum_cart < -100.0 THEN 
				e_sum_cart := -100.0; 
			END_IF
			
			Fd_cart := (Tf_cart / (Tf_cart + Ts_cart)) * Fd_prev_cart + (Kd_cart / (Tf_cart + Ts_cart)) * (e_cart - e_prev_cart);
			
			F_cart := Kp_cart * e_cart + Ki_cart * e_sum_cart + Fd_cart;
			
			e_prev_cart := e_cart;
			Fd_prev_cart := Fd_cart;
			
			// Pendulum (inner loop)
			Kp := 14.23031;
			Ki := 63.24582;
			Kd := 0.79057;
			Tf := 0.01;
			Ts := 0.002;
			
			theta_ref := -F_cart;
			theta := (I_pos_pendulum - 2048) * ((2 * amPI) / 4096);
			
			e := theta_ref - theta;
			
			e_sum := e_sum + (e * Ts);
			
			IF e_sum > 100.0 THEN 
				e_sum := 100.0; 
			ELSIF e_sum < -100.0 THEN 
				e_sum := -100.0; 
			END_IF
			
			Fd := (Tf / (Tf + Ts)) * Fd_prev + (Kd / (Tf + Ts)) * (e - e_prev);
			
			F := Kp * e + Ki * e_sum + Fd;
			
			e_prev := e;
			Fd_prev := Fd;
			
			IF (I_pos_pendulum > 1550 AND I_pos_pendulum < 2550) THEN
				Q_vel_cart := REAL_TO_INT(F * scale);
			ELSE
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
			IF (Q_vel_cart > 7500) THEN
				Q_vel_cart := 7500;
			ELSIF (Q_vel_cart < -7500) THEN
				Q_vel_cart := -7500;
			END_IF
			IF NOT I_endstop_positive OR NOT I_endstop_negative THEN
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
			
			IF (I_pos_cart > 330000 OR I_pos_cart < 50000) THEN
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
			
			IF Q_vel_cart < 0 THEN
				Q_vel_cart := Q_vel_cart - 1700;
			ELSIF Q_vel_cart > 0 THEN
				Q_vel_cart := Q_vel_cart + 1700;
			END_IF
			
		
		center: // Bruges til at bevæge til midten af båndet
			Q_vel_cart := -4000;
			IF (I_pos_cart > 190000) THEN
				systemState := idle;
			END_IF
		
		Preset: // Bruges til reset af pendul encoder
			reset_pendulum_position := TRUE;
			systemState := idle;
		
					
			
	END_CASE
	 
END_PROGRAM

PROGRAM _EXIT

	
	 
END_PROGRAM

