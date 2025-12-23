import matplotlib.pyplot as plt
import numpy as np

# Data
pts = [40, 50, 60]

# CFARNet 95% errors
cfarnet_angle = [50.4788, 0.0454, 0.0345]
cfarnet_range = [42.9059, 0.4357, 0.2516]
cfarnet_velocity = [11.1585, 0.5804, 0.3734]

# YOLO 95% errors
yolo_angle = [0.5888, 0.2197, 0.1417]
yolo_range = [1.8953, 0.5072, 0.2812]
yolo_velocity = [1.9444, 0.6373, 0.3506]

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Angle
axs[0].plot(pts, cfarnet_angle, 'o-', label='CFARNet')
axs[0].plot(pts, yolo_angle, 's-', label='YOLO')
axs[0].set_title('Angle 95% Error vs Pt')
axs[0].set_xlabel('Pt (dBm)')
axs[0].set_ylabel('Angle Error (deg)')
axs[0].legend()
axs[0].grid(True)

# Range
axs[1].plot(pts, cfarnet_range, 'o-', label='CFARNet')
axs[1].plot(pts, yolo_range, 's-', label='YOLO')
axs[1].set_title('Range 95% Error vs Pt')
axs[1].set_xlabel('Pt (dBm)')
axs[1].set_ylabel('Range Error (m)')
axs[1].legend()
axs[1].grid(True)

# Velocity
axs[2].plot(pts, cfarnet_velocity, 'o-', label='CFARNet')
axs[2].plot(pts, yolo_velocity, 's-', label='YOLO')
axs[2].set_title('Velocity 95% Error vs Pt')
axs[2].set_xlabel('Pt (dBm)')
axs[2].set_ylabel('Velocity Error (m/s)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('performance_comparison.png')
print("Plot saved to performance_comparison.png")
