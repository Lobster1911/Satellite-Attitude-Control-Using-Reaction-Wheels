import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24  # Earth's mass (kg)
R_earth = 6371e3  # Earth's radius in meters
a = 26560e3  # Semi-major axis (in meters)
eccentricity = 0.5  # Elliptical orbit eccentricity
inclination = np.radians(63.4)  # Inclination in radians

# Orbital mechanics
v_periapsis = np.sqrt(G * M * (1 + eccentricity) / (a * (1 - eccentricity)))  # Velocity at periapsis

# Initial conditions: periapsis (closest approach)
position = np.array([a * (1 - eccentricity), 0, 0])  # Starting at periapsis
velocity = np.array([0, v_periapsis * np.cos(inclination), v_periapsis * np.sin(inclination)])  # Velocity at periapsis with inclination

# Updated Inertia tensor for a cube with mass 60 kg and side length 45 cm
I = np.array([2.025, 2.025, 2.025])  # Moment of inertia for each axis (kg*m^2)

# Initial angular velocity in body frame (rad/s)
omega = np.array([0.3, 0.3, 0.3])  # Initial angular velocity of 0.5 degrees/sec

# External torque from solar radiation pressure and aerodynamic drag (combined)
external_torque = np.array([3.5e-7, 3.5e-7, 3.5e-7])  # Torque applied in Nm


# Damping factor (for satellite without reaction wheel)
damping_factor = 0.01  # Apply damping to reduce excessive growth in angular velocity

# Time and simulation parameters
dt = 1  # Time step (1 second)
time_total = 86400  # Simulate for one day (86400 seconds)
time_steps = np.arange(0, time_total, dt)

# Arrays to store data
x_pos, y_pos, z_pos = [], [], []
angular_velocity_x, angular_velocity_y, angular_velocity_z = [], [], []

# Clamp function to prevent overflow
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Maximum allowed angular velocity (to avoid overflow)
max_angular_velocity = 10.0  # Max angular velocity in rad/s
min_angular_velocity = -10.0

# Simulation loop
for t in time_steps:
    # Calculate gravitational force based on elliptical orbit dynamics
    r = np.linalg.norm(position)
    force = -G * M / r**2 * (position / r)

    # Update velocity and position using Newtonian mechanics
    acceleration = force
    velocity += acceleration * dt
    position += velocity * dt

    # Store position data
    x_pos.append(position[0])
    y_pos.append(position[1])
    z_pos.append(position[2])

    # Apply Euler's equations to update angular velocity with external torque
    domega1 = (external_torque[0] - (I[2] - I[1]) * omega[1] * omega[2]) / I[0]
    domega2 = (external_torque[1] - (I[0] - I[2]) * omega[0] * omega[2]) / I[1]
    domega3 = (external_torque[2] - (I[1] - I[0]) * omega[0] * omega[1]) / I[2]

    # Update angular velocities with clamping to prevent overflow
    omega[0] = clamp(omega[0] + domega1 * dt, min_angular_velocity, max_angular_velocity)
    omega[1] = clamp(omega[1] + domega2 * dt, min_angular_velocity, max_angular_velocity)
    omega[2] = clamp(omega[2] + domega3 * dt, min_angular_velocity, max_angular_velocity)

    # Store angular velocities for plotting
    angular_velocity_x.append(omega[0])
    angular_velocity_y.append(omega[1])
    angular_velocity_z.append(omega[2])

# Plot the orbit in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_pos, y_pos, z_pos)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Satellite Elliptical Orbit with Inclination (3D)")
plt.show()

# Plot angular velocity over time
plt.figure()
plt.plot(time_steps, angular_velocity_x, label="ω_x")
plt.plot(time_steps, angular_velocity_y, label="ω_y")
plt.plot(time_steps, angular_velocity_z, label="ω_z")
plt.xlabel("Time (seconds)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity without Reaction Wheel Control (Reduced Torque)")
plt.legend()
plt.grid(True)
plt.show()

# After your simulation loop, save position and angular velocity data to CSV
with open('reduced_torque_no_reaction_wheel_3d.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "x", "y", "z", "omega_x", "omega_y", "omega_z"])  # Header row
    for i in range(len(time_steps)):
        writer.writerow([time_steps[i], x_pos[i], y_pos[i], z_pos[i], 
                         angular_velocity_x[i], angular_velocity_y[i], angular_velocity_z[i]])
