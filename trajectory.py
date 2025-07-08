import numpy as np

def generate_trajectory(total_time, dt, theta_min_deg, theta_max_deg, r_min, r_max, min_speed, max_speed, sector_idx=None, total_sectors=3, random_flag=0, circle_mode=True):
    """
    生成目标运动轨迹

    输入参数：
      total_time     - 总运动时间（秒）
      dt             - 时间步长（秒）
      theta_min_deg  - 初始角度下限（单位：度）
      theta_max_deg  - 初始角度上限（单位：度）
      r_min, r_max   - 初始距离范围（单位：米）
      min_speed, max_speed - 速度范围（单位：km/h，函数内部转换为 m/s）
      sector_idx     - 保留参数，不再使用
      total_sectors  - 保留参数，不再使用
      random_flag    - 轨迹生成模式（0：确保角度差，1：完全随机）
      circle_mode    - 是否使用固定圆形区域模式

    固定圆形区域参数（当circle_mode=True时使用）：
      圆心坐标：(50, 50)，半径30m
      最小角度差：1.5度

    输出参数：
      x, y         - 目标在二维平面上的位置序列（1D 数组）
      vx_vec, vy_vec - x 和 y 方向的速度（1D 数组）
      vr           - 径向速度分量（目标靠近或远离雷达的速度，1D 数组）
      vt           - 切向速度分量（垂直于径向的速度，1D 数组）
      r_vals       - 目标与原点的径向距离（1D 数组）
      theta_vals   - 目标位置角度（度）
    """
    # 计算时间步数
    N = int(np.floor(total_time / dt))
    t = np.linspace(0, total_time, N)  # 时间向量

    # 初始化各个变量
    x = np.zeros(N)
    y = np.zeros(N)
    vx_vec = np.zeros(N)
    vy_vec = np.zeros(N)
    vr = np.zeros(N)
    vt = np.zeros(N)
    r_vals = np.zeros(N)
    theta_vals = np.zeros(N)

    # 定义安全边界和最小角度差
    theta_safe_min = -60
    theta_safe_max = 60
    min_angle_diff = 10  # 最小角度差（度），从15度改为1.5度
    
    # 定义固定圆形区域
    if circle_mode:
        # 单一圆形区域：圆心在(50, 50)，半径30m
        circle = {"x": 50, "y": 0, "radius": 30}
        circles = [circle]
    
    # 根据模式生成初始位置
    if circle_mode:
        # 使用固定圆形区域模式
        # 如果有角度限制，确保各目标间角度差大于min_angle_diff
        if random_flag == 0 and hasattr(generate_trajectory, 'chosen_angles'):
            # 已经选择了一些角度，需要确保新角度与已有角度差大于min_angle_diff
            valid_position = False
            max_attempts = 100
            attempt = 0
            
            while not valid_position and attempt < max_attempts:
                # 在圆内随机选择一个点 (确保面积上均匀分布)
                # 1. 为 r^2 生成一个在 [0, 1) 范围内的随机比例因子
                random_proportion_for_r_squared = np.random.rand()
                # 2. 计算圆内随机点的 r^2 值
                r_squared_in_circle = circle["radius"]**2 * random_proportion_for_r_squared
                # 3. 开根号得到 r 值
                r_in_circle = np.sqrt(r_squared_in_circle)
                theta_in_circle = 2 * np.pi * np.random.rand()
                
                # 计算在全局坐标系中的位置
                x_pos = circle["x"] + r_in_circle * np.cos(theta_in_circle)
                y_pos = circle["y"] + r_in_circle * np.sin(theta_in_circle)
                
                # 计算极坐标
                r0 = np.sqrt(x_pos**2 + y_pos**2)
                theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
                
                # 检查与已选角度的差值
                if all(abs(theta0_deg - angle) >= min_angle_diff for angle in generate_trajectory.chosen_angles):
                    valid_position = True
                    generate_trajectory.chosen_angles.append(theta0_deg)
                    break
                
                attempt += 1
            
            # 如果无法找到满足条件的角度，选择最接近的可行角度
            if not valid_position:
                closest_angle = min(generate_trajectory.chosen_angles, key=lambda x: abs(x - theta0_deg))
                theta0_deg = closest_angle + min_angle_diff if theta0_deg > closest_angle else closest_angle - min_angle_diff
                
                # 在圆边缘找一个接近目标角度的点
                x_pos = circle["x"] + circle["radius"] * np.cos(np.deg2rad(theta0_deg))
                y_pos = circle["y"] + circle["radius"] * np.sin(np.deg2rad(theta0_deg))
                
                r0 = np.sqrt(x_pos**2 + y_pos**2)
                theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
                generate_trajectory.chosen_angles.append(theta0_deg)
        else:
            # 第一次调用或不需要角度限制，在圆内随机选择一个点 (确保面积上均匀分布)
            # 1. 为 r^2 生成一个在 [0, 1) 范围内的随机比例因子
            random_proportion_for_r_squared = np.random.rand()
            # 2. 计算圆内随机点的 r^2 值
            r_squared_in_circle = circle["radius"]**2 * random_proportion_for_r_squared
            # 3. 开根号得到 r 值
            r_in_circle = np.sqrt(r_squared_in_circle)
            theta_in_circle = 2 * np.pi * np.random.rand()
            
            # 计算在全局坐标系中的位置
            x_pos = circle["x"] + r_in_circle * np.cos(theta_in_circle)
            y_pos = circle["y"] + r_in_circle * np.sin(theta_in_circle)
            
            # 计算极坐标
            r0 = np.sqrt(x_pos**2 + y_pos**2)
            theta0_deg = np.rad2deg(np.arctan2(y_pos, x_pos))
            
            # 初始化或更新已选角度列表
            if hasattr(generate_trajectory, 'chosen_angles'):
                generate_trajectory.chosen_angles.append(theta0_deg)
            else:
                generate_trajectory.chosen_angles = [theta0_deg]
    else:
        # 原始方式：根据random_flag确定初始角度
        if random_flag == 1:
            # 完全随机模式：随机选择角度
            theta0_deg = theta_min_deg + (theta_max_deg - theta_min_deg) * np.random.rand()
        else:
            # 确保角度差模式：在安全范围内随机选择角度
            # 如果之前已经选择了角度，确保新角度与已选角度的差大于min_angle_diff
            if hasattr(generate_trajectory, 'chosen_angles'):
                max_attempts = 100  # 最大尝试次数
                attempt = 0
                while attempt < max_attempts:
                    theta0_deg = theta_safe_min + (theta_safe_max - theta_safe_min) * np.random.rand()
                    # 检查与所有已选角度的差
                    if all(abs(theta0_deg - angle) >= min_angle_diff for angle in generate_trajectory.chosen_angles):
                        generate_trajectory.chosen_angles.append(theta0_deg)
                        break
                    attempt += 1
                if attempt == max_attempts:
                    # 如果无法找到满足条件的角度，选择与最近角度相差min_angle_diff的角度
                    closest_angle = min(generate_trajectory.chosen_angles, key=lambda x: abs(x - theta0_deg))
                    theta0_deg = closest_angle + min_angle_diff if theta0_deg > closest_angle else closest_angle - min_angle_diff
                    # 确保角度在安全范围内
                    theta0_deg = np.clip(theta0_deg, theta_safe_min, theta_safe_max)
                    generate_trajectory.chosen_angles.append(theta0_deg)
            else:
                # 第一次调用，直接随机选择角度
                theta0_deg = theta_safe_min + (theta_safe_max - theta_safe_min) * np.random.rand()
                generate_trajectory.chosen_angles = [theta0_deg]

        # 随机选择初始距离 (确保在 [r_min, r_max]定义的环形区域内面积上均匀分布)
        # 1. 为 r^2 生成一个在 [0, 1) 范围内的随机比例因子
        random_proportion_for_r_squared = np.random.rand()
        # 2. 计算在 [r_min^2, r_max^2) 范围内的 r^2 值
        r_squared = r_min**2 + (r_max**2 - r_min**2) * random_proportion_for_r_squared
        # 3. 开根号得到 r0 值
        r0 = np.sqrt(r_squared)
        
        # 设置初始位置
        x_pos = r0 * np.cos(np.deg2rad(theta0_deg))
        y_pos = r0 * np.sin(np.deg2rad(theta0_deg))

    # 速度设置（完全随机）
    speed_kmh = min_speed + (max_speed - min_speed) * np.random.rand()
    
    # 转换为m/s并随机决定接近/远离方向
    radial_speed = (speed_kmh / 3.6) * (2 * (np.random.rand() > 0.5) - 1)
    
    # 设置初始位置
    x[0] = x_pos
    y[0] = y_pos
    
    # 设置初始速度
    vr[0] = radial_speed
    vt[0] = 0  # 切向速度恒为0
    vx_vec[0] = radial_speed * np.cos(np.deg2rad(theta0_deg))
    vy_vec[0] = radial_speed * np.sin(np.deg2rad(theta0_deg))

    # 沿径向方向运动：位置更新
    for i in range(1, N):
        # 更新位置
        x[i] = x[i-1] + vx_vec[i-1] * dt
        y[i] = y[i-1] + vy_vec[i-1] * dt
        
        # 计算新的径向距离和角度
        r_vals[i] = np.sqrt(x[i]**2 + y[i]**2)
        theta_vals[i] = np.rad2deg(np.arctan2(y[i], x[i]))
        
        # 更新速度分量（保持径向运动）
        vx_vec[i] = radial_speed * np.cos(np.deg2rad(theta_vals[i]))
        vy_vec[i] = radial_speed * np.sin(np.deg2rad(theta_vals[i]))
        vr[i] = radial_speed  # 径向速度保持不变
        vt[i] = 0             # 切向速度为零

    # 计算完整的目标与原点的径向距离，防止出现零距离
    r_vals = np.sqrt(x**2 + y**2)
    r_vals[r_vals == 0] = 1e-12
    
    # 计算位置角度（度数）
    theta_vals = np.rad2deg(np.arctan2(y, x))

    return x, y, vx_vec, vy_vec, vr, vt, r_vals, theta_vals
