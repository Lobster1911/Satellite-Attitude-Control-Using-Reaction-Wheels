import numpy as np
import matplotlib.pyplot as plt
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

# Updated Inertia tensor (based on satellite dimensions and mass)
I = np.array([2.025, 2.025, 2.025])  # Moment of inertia for each axis (kg*m^2)

# Initial angular velocity in body frame (rad/s)
omega = np.array([0.3, 0.3, 0.3])  # Small initial angular velocity

# External torque from solar radiation pressure and aerodynamic drag (combined)
external_torque = np.array([3.5e-7, 3.5e-7, 3.5e-7]) # Simulate small disturbance torques in Nm

# Reaction wheel control (PID controller parameters)
k_p = 0.1  # Proportional gain
k_i = 0.01  # Integral gain
k_d = 0.2  # Derivative gain

# PID controller state variables
integral_error = np.array([0.0, 0.0, 0.0])  # Integral of the angular velocity
previous_omega = omega.copy()  # To calculate the derivative

# Damping factor
damping_factor = 0.01  # Apply damping to reduce excessive oscillation

# Time and simulation parameters
dt = 1  # Time step (1 second)
time_total = 86400  # Simulate for one day (86400 seconds)
time_steps = np.arange(0, time_total, dt)

# Arrays to store data
x_pos, y_pos, z_pos = [], [], []
angular_velocity_x, angular_velocity_y, angular_velocity_z = [], [], []
reaction_wheel_torque_x, reaction_wheel_torque_y, reaction_wheel_torque_z = [], [], []

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

    # PID control for reaction wheels
    error = -omega  # We want to bring the angular velocity to zero (stabilization)

    # Proportional term
    proportional_torque = k_p * error

    # Integral term (accumulating error over time)
    integral_error += error * dt
    integral_torque = k_i * integral_error

    # Derivative term (rate of change of angular velocity)
    derivative_torque = k_d * (omega - previous_omega) / dt
    previous_omega = omega.copy()

    # Total reaction wheel torque (PID)
    reaction_wheel_torque = proportional_torque + integral_torque + derivative_torque

    # Apply damping to reaction wheel torque to prevent excessive oscillation
    reaction_wheel_torque -= damping_factor * omega

    # Apply Euler's equations to update angular velocity with external and reaction wheel torque
    domega1 = (external_torque[0] + reaction_wheel_torque[0] - (I[2] - I[1]) * omega[1] * omega[2]) / I[0]
    domega2 = (external_torque[1] + reaction_wheel_torque[1] - (I[0] - I[2]) * omega[0] * omega[2]) / I[1]
    domega3 = (external_torque[2] + reaction_wheel_torque[2] - (I[1] - I[0]) * omega[0] * omega[1]) / I[2]

    # Update angular velocities with clamping to prevent overflow
    omega[0] = clamp(omega[0] + domega1 * dt, min_angular_velocity, max_angular_velocity)
    omega[1] = clamp(omega[1] + domega2 * dt, min_angular_velocity, max_angular_velocity)
    omega[2] = clamp(omega[2] + domega3 * dt, min_angular_velocity, max_angular_velocity)

    # Store angular velocities and reaction wheel torques for plotting
    angular_velocity_x.append(omega[0])
    angular_velocity_y.append(omega[1])
    angular_velocity_z.append(omega[2])
    
    reaction_wheel_torque_x.append(reaction_wheel_torque[0])
    reaction_wheel_torque_y.append(reaction_wheel_torque[1])
    reaction_wheel_torque_z.append(reaction_wheel_torque[2])

# Plot angular velocity over time
plt.figure()
plt.plot(time_steps, angular_velocity_x, label="ω_x")
plt.plot(time_steps, angular_velocity_y, label="ω_y")
plt.plot(time_steps, angular_velocity_z, label="ω_z")
plt.xlabel("Time (seconds)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity with Reaction Wheel Control (PID)")
plt.legend()
plt.grid(True)
plt.show()

# Plot reaction wheel torques over time
plt.figure()
plt.plot(time_steps, reaction_wheel_torque_x, label="Torque_x")
plt.plot(time_steps, reaction_wheel_torque_y, label="Torque_y")
plt.plot(time_steps, reaction_wheel_torque_z, label="Torque_z")
plt.xlabel("Time (seconds)")
plt.ylabel("Reaction Wheel Torque (Nm)")
plt.title("Reaction Wheel Torques Over Time")
plt.legend()
plt.grid(True)
plt.show()

# After your simulation loop, save position and angular velocity data to CSV
with open('reaction_wheel_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "omega_x", "omega_y", "omega_z", "torque_x", "torque_y", "torque_z"])  # Header row
    for i in range(len(time_steps)):
        writer.writerow([time_steps[i], angular_velocity_x[i], angular_velocity_y[i], angular_velocity_z[i], 
                         reaction_wheel_torque_x[i], reaction_wheel_torque_y[i], reaction_wheel_torque_z[i]])