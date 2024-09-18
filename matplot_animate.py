import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# Initialize lists to store the data from the CSV
time_steps = []
x_pos = []
y_pos = []
z_pos = []
angular_velocity_x = []
angular_velocity_y = []
angular_velocity_z = []

# Read the data from the CSV file
with open('satellite_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        time_steps.append(float(row[0]))
        x_pos.append(float(row[1]))
        y_pos.append(float(row[2]))
        z_pos.append(float(row[3]))
        angular_velocity_x.append(float(row[4]))
        angular_velocity_y.append(float(row[5]))
        angular_velocity_z.append(float(row[6]))

# Constants for visualization
R_earth = 6371e3  # Earth's radius in meters
a = 26560e3  # Semi-major axis for scaling (already defined in your simulation)

# Create figure and 3D axis for orbit
fig = plt.figure(figsize=(12, 6))
ax_orbit = fig.add_subplot(121, projection='3d')  # 3D plot for satellite orbit

# Initial plot settings for the orbit
ax_orbit.set_xlim([-1.5 * a, 1.5 * a])
ax_orbit.set_ylim([-1.5 * a, 1.5 * a])
ax_orbit.set_zlim([-1.5 * a, 1.5 * a])
ax_orbit.set_xlabel("X Position (m)")
ax_orbit.set_ylabel("Y Position (m)")
ax_orbit.set_zlabel("Z Position (m)")
ax_orbit.set_title("Satellite Orbit with Path (LVLH Frame)")

# Draw Earth as a blue sphere for reference
u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
x_sphere = R_earth * np.cos(u) * np.sin(v)
y_sphere = R_earth * np.sin(u) * np.sin(v)
z_sphere = R_earth * np.cos(v)
ax_orbit.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.3)

# Plot satellite's initial position and path
satellite, = ax_orbit.plot([], [], [], 'ro', markersize=8)
orbit_path, = ax_orbit.plot([], [], [], 'g-', linewidth=2)  # Plot for the satellite's path

# Function to update the satellite's 3D position and path
def update_orbit(frame):
    # Update satellite's 3D position using set_data() and set_3d_properties()
    satellite.set_data([x_pos[frame]], [y_pos[frame]])  # Update current position
    satellite.set_3d_properties([z_pos[frame]])  # Update the z coordinate
    
    # Update the path (showing all positions up to the current frame)
    orbit_path.set_data(x_pos[:frame], y_pos[:frame])
    orbit_path.set_3d_properties(z_pos[:frame])
    
    return satellite, orbit_path

# Create the animation for the orbit
ani_orbit = FuncAnimation(fig, update_orbit, frames=len(time_steps), interval=0.9, blit=True)

# Plot angular velocities over time on a separate subplot
ax_velocity = fig.add_subplot(122)  # 2D plot for angular velocity

# Plot angular velocity over time
ax_velocity.plot(time_steps, angular_velocity_x, label="ω_x (rad/s)")
ax_velocity.plot(time_steps, angular_velocity_y, label="ω_y (rad/s)")
ax_velocity.plot(time_steps, angular_velocity_z, label="ω_z (rad/s)")
ax_velocity.set_xlabel("Time (seconds)")
ax_velocity.set_ylabel("Angular Velocity (rad/s)")
ax_velocity.set_title("Angular Velocity Altered by Reaction Wheels")
ax_velocity.legend()
ax_velocity.grid(True)

# Show both plots
plt.tight_layout()
plt.show()
