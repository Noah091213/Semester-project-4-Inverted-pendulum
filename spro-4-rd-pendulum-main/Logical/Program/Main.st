
PROGRAM _INIT

	Q_vel_cart := 0;
	I_pos_pendulum;
	reset_pendulum_position;
	I_pos_cart;
	reset_cart_position;
	systemState := idle;
	
	scale := 2275;
	
END_PROGRAM

PROGRAM _CYCLIC

	// Handle EStop
	IF(SI_estop_A AND SI_estop_B) THEN
		Q_drive_enable := TRUE;
	ELSE 
		Q_drive_enable  := FALSE;
		Q_reset_LED := FALSE;
		systemState := idle;
	END_IF
	
//	// Auto Reset Btn
//	IF (NOT I_drive_no_fault) THEN
//		Q_reset_LED := FALSE;
//		Q_drive_faultreset := NOT Q_drive_faultreset;
//	ELSE
//		Q_reset_LED := TRUE;
//	END_IF
	
	// Handle Reset Btn
	IF(SI_reset_btn) THEN
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
			
			// Cascade reset
			referenceNext_cart := 0;
			positionNext_cart := 0;
			error_cart := 0;
			errorNext_cart := 0;
			outputNextI_cart := 0;
			outputNextD_cart := 0;
			outputD_cart := 0;
			outputNext_cart := 0;
			output_cart := 0;

			angleNext_pen := 0;
			errorNext_pen := 0;
			error_pen := 0;
			outputNextI_pen := 0;
			outputNextD_pen := 0;
			outputD_pen := 0;
			outputNext_pen := 0;
			output_pen := 0;
			
			// LQR reset
			angle_pen := 0;
			position_cart := 0;
			
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
			
		swing: 
			
		cascade:
			// Initial values - Ideally set these once, not every cycle
			referenceNext_cart := 191000.0 / 243500.0; 

			Kp_cart := -0.117;    // Adjusted to match your working "x50" gain
			Ki_cart := 0.0;
			Kd_cart := -0.234;    // Adjusted to match your working "x50" gain
			Tf_cart := 0.01;      // Filter time constant

			Kp_pen := 14.23031;
			Ki_pen := 63.24582;
			Kd_pen := 0.79057;
			Tf_pen := 0.01;       // Filter time constant

			T := 0.001;           // Sample time

			// --- CART (Outer Loop) ---
			positionNext_cart := I_pos_cart / 243500.0;
			errorNext_cart := referenceNext_cart - positionNext_cart;

			// Integral with T
			outputNextI_cart := outputNextI_cart + Ki_cart * T * errorNext_cart;

			// Filtered Derivative
			outputNextD_cart := (Tf_cart / (Tf_cart + T)) * outputD_cart + (Kd_cart / (Tf_cart + T)) * (errorNext_cart - error_cart);

			outputNext_cart := Kp_cart * errorNext_cart + outputNextI_cart + outputNextD_cart;

			// Save state for next cycle
			error_cart := errorNext_cart;
			outputD_cart := outputNextD_cart;


			// --- PENDULUM (Inner Loop) ---
			angleNext_pen := (I_pos_pendulum - 2048) * ((2 * amPI) / 4096);

			// The Cart output defines the required tilt angle
			errorNext_pen := -outputNext_cart - angleNext_pen;

			// Integral with Anti-Windup
			outputNextI_pen := outputNextI_pen + Ki_pen * T * errorNext_pen;
			IF outputNextI_pen > 5.0 THEN outputNextI_pen := 5.0; ELSIF outputNextI_pen < -5.0 THEN outputNextI_pen := -5.0; END_IF

			// Filtered Derivative
			outputNextD_pen := (Tf_pen / (Tf_pen + T)) * outputD_pen + (Kd_pen / (Tf_pen + T)) * (errorNext_pen - error_pen);

			outputNext_pen := Kp_pen * errorNext_pen + outputNextI_pen + outputNextD_pen;

			// Save state for next cycle
			error_pen := errorNext_pen;
			outputD_pen := outputNextD_pen;
			
			IF (I_pos_pendulum > 1550 AND I_pos_pendulum < 2550) THEN
				Q_vel_cart := REAL_TO_INT(outputNext_pen * scale);
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
			
//			IF Q_vel_cart < 0 THEN
//				Q_vel_cart := Q_vel_cart - 1000;
//			ELSIF Q_vel_cart > 0 THEN
//				Q_vel_cart := Q_vel_cart + 1000;
//			END_IF
			
		lqr:
			
			K1 :=  -3.1058;
			K2 := -11.5401;
			K3 :=  33.7288;
			K4 :=   5.1146;
			
			positionNext_cart := (I_pos_cart - 191000) / 243500.0;
			velocity_cart := (((I_pos_cart - 191000) - position_cart ) / 243500.0 ) / 0.001;
			
			angleNext_pen := (I_pos_pendulum - 2048) * ((2 * amPI) / 4096);
			velocity_pen := (((I_pos_pendulum - 2048) - angle_pen ) * ((2 * amPI) / 4096)) / 0.001;
			
			force := -( K1 * positionNext_cart ) + ( K2 * velocity_cart ) + ( K3 * angleNext_pen ) + ( K4 * velocity_pen );
			
			position_cart := positionNext_cart;
			angle_pen := angleNext_pen;
			
			IF (I_pos_pendulum > 1550 AND I_pos_pendulum < 2550) THEN
				Q_vel_cart := REAL_TO_INT(force * scale);
			ELSE
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
			IF (Q_vel_cart > 12000) THEN
				Q_vel_cart := 12000;
			ELSIF (Q_vel_cart < -12000) THEN
				Q_vel_cart := -12000;
			END_IF
			IF NOT I_endstop_positive OR NOT I_endstop_negative THEN
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
			
			IF (I_pos_cart > 330000 OR I_pos_cart < 50000) THEN
				Q_vel_cart := 0;
				systemState := idle;
			END_IF
		
		center:
			Q_vel_cart := -4000;
			IF (I_pos_cart > 191000) THEN
				systemState := idle;
			END_IF
		
		pendulReset:
			reset_pendulum_position := TRUE;
			systemState := idle;
						
	END_CASE
	 
END_PROGRAM

PROGRAM _EXIT

	
	 
END_PROGRAM

