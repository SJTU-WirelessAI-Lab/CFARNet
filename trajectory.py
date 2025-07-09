import numpy as np

def generate_trajectory(total_time, dt, theta_min_deg, theta_max_deg, r_min, r_max, min_speed, max_speed, sector_idx=None, total_sectors=3, random_flag=0, circle_mode=True):
    """
    Generate target movement trajectory

    Input parameters:
      total_time     - Total movement time (seconds)
      dt             - Time step size (seconds)
      theta_min_deg  - Initial angle lower bound (degrees)
      theta_max_deg  - Initial angle upper bound (degrees)
      r_min, r_max   - Initial distance range (meters)
      min_speed, max_speed - Speed range (km/h, converted to m/s internally)
      sector_idx     - Reserved parameter, no longer used
      total_sectors  - Reserved parameter, no longer used
      random_flag    - Trajectory generation mode (0: ensure angle difference, 1: completely random)
      circle_mode    - Whether to use fixed circular area mode

    Fixed circular area parameters (used when circle_mode=True):
      Circle center: (50, 50), radius 30m
      Minimum angle difference: 1.5 degrees

    Output parameters:
      x, y         - Target position sequence in 2D plane (1D arrays)
      vx_vec, vy_vec - Velocities in x and y directions (1D arrays)
      vr           - Radial velocity component (target approaching/receding radar velocity, 1D array)
      vt           - Tangential velocity component (velocity perpendicular to radial, 1D array)
      r_vals       - Target radial distance from origin (1D array)
      theta_vals   - Target position angle (degrees)
    """
    # Calculate number of time steps
    N = int(np.floor(total_time / dt))
    t = np.linspace(0, total_time, N)  # Time vector

    # Initialize all variables
    x = np.zeros(N)
    y = np.zeros(N)
    vx_vec = np.zeros(N)
    vy_vec = np.zeros(N)
    vr = np.zeros(N)
    vt = np.zeros(N)
    r_vals = np.zeros(N)
    theta_vals = np.zeros(N)

    # Define safe boundaries and minimum angle difference
    theta_safe_min = -60
    theta_safe_max = 60
    min_angle_diff = 10  # Minimum angle difference (degrees), changed from 15 to 1.5 degrees
    
    # Define fixed circular area
    if circle_mode:
        # Single circular area: center at (50, 50), radius 30m
        circle = {"x": 50, "y": 0, "radius": 30}
        circles = [circle]
    
    # Generate initial position based on mode
    if circle_mode:
        # Use fixed circular area mode
        # If angle constraints exist, ensure new angle differs from existing angles by more than min_angle_diff
        if random_flag == 0 and hasattr(generate_trajectory, 'chosen_angles'):
            # Some angles already chosen, need to ensure new angle differs from existing ones by more than min_angle_diff
            valid_position = False
            max_attempts = 100
            attempt = 0
            
            while not valid_position and attempt < max_attempts:
                # Randomly select a point within circle (ensure uniform distribution by area)
                # 1. Generate random proportion factor for r^2 in range [0, 1)
                random_proportion_for_r_squared = np.random.rand()
                # 2. Calculate r^2 value for random point within circle
                r_squared_in_circle = circle["radius"]**2 * random_proportion_for_r_squared
                # 3. Take square root to get r value
                r_in_circle = np.sqrt(r_squared_in_circle)
                theta_in_circle = 2 * np.pi * np.random.rand()
                
                # Calculate position in global coordinate system
                x_pos = circle["x"] + r_in_circle * np.cos(theta_in_circle)
                y_pos = circle["y"] + r_in_circle * np.sin(theta_in_circle)
                
                # Calculate polar coordinates
                r0 = np.sqrt(x_pos**2 + y_pos**2)
                theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
                
                # Check difference with already chosen angles
                if all(abs(theta0_deg - angle) >= min_angle_diff for angle in generate_trajectory.chosen_angles):
                    valid_position = True
                    generate_trajectory.chosen_angles.append(theta0_deg)
                    break
                
                attempt += 1
            
            # If cannot find satisfactory angle, choose closest feasible angle
            if not valid_position:
                closest_angle = min(generate_trajectory.chosen_angles, key=lambda x: abs(x - theta0_deg))
                theta0_deg = closest_angle + min_angle_diff if theta0_deg > closest_angle else closest_angle - min_angle_diff
                
                # Find a point near target angle on circle edge
                x_pos = circle["x"] + circle["radius"] * np.cos(np.deg2rad(theta0_deg))
                y_pos = circle["y"] + circle["radius"] * np.sin(np.deg2rad(theta0_deg))
                
                r0 = np.sqrt(x_pos**2 + y_pos**2)
                theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
                generate_trajectory.chosen_angles.append(theta0_deg)
        else:
            # First call or no angle constraints needed, randomly select point within circle (ensure uniform distribution by area)
            # 1. Generate random proportion factor for r^2 in range [0, 1)
            random_proportion_for_r_squared = np.random.rand()
            # 2. Calculate r^2 value for random point within circle
            r_squared_in_circle = circle["radius"]**2 * random_proportion_for_r_squared
            # 3. Take square root to get r value
            r_in_circle = np.sqrt(r_squared_in_circle)
            theta_in_circle = 2 * np.pi * np.random.rand()
            
            # Calculate position in global coordinate system
            x_pos = circle["x"] + r_in_circle * np.cos(theta_in_circle)
            y_pos = circle["y"] + r_in_circle * np.sin(theta_in_circle)
            
            # Calculate polar coordinates
            r0 = np.sqrt(x_pos**2 + y_pos**2)
            theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
            
            # Initialize or update chosen angles list
            if hasattr(generate_trajectory, 'chosen_angles'):
                generate_trajectory.chosen_angles.append(theta0_deg)
            else:
                generate_trajectory.chosen_angles = [theta0_deg]
    else:
        # Original method: determine initial angle based on random_flag
        if random_flag == 1:
            # Completely random mode: randomly choose angle
            theta0_deg = theta_min_deg + (theta_max_deg - theta_min_deg) * np.random.rand()
        else:
            # Ensure angle difference mode: randomly choose angle within safe range
            # If angles were previously chosen, ensure new angle differs from chosen angles by more than min_angle_diff
            if hasattr(generate_trajectory, 'chosen_angles'):
                max_attempts = 100  # Maximum number of attempts
                attempt = 0
                while attempt < max_attempts:
                    theta0_deg = theta_safe_min + (theta_safe_max - theta_safe_min) * np.random.rand()
                    # Check difference with all chosen angles
                    if all(abs(theta0_deg - angle) >= min_angle_diff for angle in generate_trajectory.chosen_angles):
                        generate_trajectory.chosen_angles.append(theta0_deg)
                        break
                    attempt += 1
                if attempt == max_attempts:
                    # If cannot find satisfactory angle, choose angle differing from nearest angle by min_angle_diff
                    closest_angle = min(generate_trajectory.chosen_angles, key=lambda x: abs(x - theta0_deg))
                    theta0_deg = closest_angle + min_angle_diff if theta0_deg > closest_angle else closest_angle - min_angle_diff
                    # Ensure angle is within safe range
                    theta0_deg = np.clip(theta0_deg, theta_safe_min, theta_safe_max)
                    generate_trajectory.chosen_angles.append(theta0_deg)
            else:
                # First call, directly choose random angle
                theta0_deg = theta_safe_min + (theta_safe_max - theta_safe_min) * np.random.rand()
                generate_trajectory.chosen_angles = [theta0_deg]

        # Randomly select initial distance (ensure uniform distribution by area within annular region defined by [r_min, r_max])
        # 1. Generate random proportion factor for r^2 in range [0, 1)
        random_proportion_for_r_squared = np.random.rand()
        # 2. Calculate r^2 value in range [r_min^2, r_max^2)
        r_squared = r_min**2 + (r_max**2 - r_min**2) * random_proportion_for_r_squared
        # 3. Take square root to get r0 value
        r0 = np.sqrt(r_squared)
        
        # Set initial position
        x_pos = r0 * np.cos(np.deg2rad(theta0_deg))
        y_pos = r0 * np.sin(np.deg2rad(theta0_deg))

    # Velocity setting (completely random)
    speed_kmh = min_speed + (max_speed - min_speed) * np.random.rand()
    
    # Convert to m/s and randomly decide approach/recede direction
    radial_speed = (speed_kmh / 3.6) * (2 * (np.random.rand() > 0.5) - 1)
    
    # Set initial position
    x[0] = x_pos
    y[0] = y_pos
    
    # Set initial velocity
    vr[0] = radial_speed
    vt[0] = 0  # Tangential velocity always 0
    vx_vec[0] = radial_speed * np.cos(np.deg2rad(theta0_deg))
    vy_vec[0] = radial_speed * np.sin(np.deg2rad(theta0_deg))

    # Motion along radial direction: position update
    for i in range(1, N):
        # Update position
        x[i] = x[i-1] + vx_vec[i-1] * dt
        y[i] = y[i-1] + vy_vec[i-1] * dt
        
        # Calculate new radial distance and angle
        r_vals[i] = np.sqrt(x[i]**2 + y[i]**2)
        theta_vals[i] = np.rad2deg(np.arctan2(y[i], x[i]))
        
        # Update velocity components (maintain radial motion)
        vx_vec[i] = radial_speed * np.cos(np.deg2rad(theta_vals[i]))
        vy_vec[i] = radial_speed * np.sin(np.deg2rad(theta_vals[i]))
        vr[i] = radial_speed  # Radial velocity remains constant
        vt[i] = 0             # Tangential velocity is zero

    # Calculate complete target radial distance from origin, prevent zero distance
    r_vals = np.sqrt(x**2 + y**2)
    r_vals[r_vals == 0] = 1e-12
    
    # Calculate position angle (degrees)
    theta_vals = np.rad2deg(np.arctan2(y, x))

    return x, y, vx_vec, vy_vec, vr, vt, r_vals, theta_vals
