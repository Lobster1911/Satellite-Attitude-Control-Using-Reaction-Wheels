# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from vpython import canvas, box, vector, rate, color

# Define PID controller class
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.dt = dt  # Time step
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)

    def compute_control(self, error):
        # Calculate proportional, integral, and derivative terms
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error

        # PID formula
        control_signal = (self.Kp * error +
                          self.Ki * self.integral_error +
                          self.Kd * derivative_error)
        return control_signal

# Satellite properties
I_satellite = np.diag([10, 15, 20])  # Moment of inertia of satellite (kg·m^2) along x, y, z axes
torque_wheel = np.zeros(3)  # Initial torque applied by reaction wheels
desired_orientation = np.array([0, 0, 0])  # Desired orientation (rad)

# Time array for simulation
dt = 0.01  # Time step
t = np.arange(0, 50, dt)  # Simulate for 50 seconds

# PID controller parameters
Kp = np.array([50, 40, 30])  # Proportional gains for x, y, z
Ki = np.array([0.1, 0.1, 0.1])  # Integral gains for x, y, z
Kd = np.array([20, 15, 10])  # Derivative gains for x, y, z
pid = PIDController(Kp, Ki, Kd, dt)

# Initial conditions: [omega_x, omega_y, omega_z] (initial angular velocity of satellite)
initial_conditions = [0.1, -0.05, 0.2]  # Initial angular velocities

# System dynamics: Satellite + PID control
def satellite_dynamics_with_control(omega, t, I_satellite, torque_wheel, pid, desired_orientation):
    # Current orientation error (proportional to angular velocity here for simplicity)
    error = desired_orientation - omega

    # PID controller output for reaction wheel torques
    control_torque = pid.compute_control(error)

    # Angular accelerations using Euler's equations (reaction wheel torque applied)
    d_omega = np.linalg.inv(I_satellite).dot(control_torque - np.cross(omega, I_satellite.dot(omega)))
    return d_omega

# Integrating satellite dynamics with control
omega_solution = odeint(satellite_dynamics_with_control, initial_conditions, t, args=(I_satellite, torque_wheel, pid, desired_orientation))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, omega_solution[:, 0], label='Omega_x (PID Controlled)')
plt.plot(t, omega_solution[:, 1], label='Omega_y (PID Controlled)')
plt.plot(t, omega_solution[:, 2], label='Omega_z (PID Controlled)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('3D Satellite Angular Velocities with PID Control (X, Y, Z Axes)')
plt.legend()
plt.grid(True)
plt.show()




# Initialize the VPython canvas for 3D visualization
scene = canvas(title="Satellite Attitude Control Visualization", width=800, height=600)

# Create a box to represent the satellite
satellite = box(pos=vector(0, 0, 0), size=vector(2, 1, 1), color=color.cyan)

# Use the previously computed angular velocities (omega_solution) from the PID controller
# For now, I am using random angular velocities for demonstration; replace it with omega_solution
angular_velocities = np.random.rand(1000, 3)  # Replace this with omega_solution
dt = 0.01  # Time step for VPython visualization

# Main loop for 3D visualization
for i in range(len(angular_velocities)):
    rate(100)  # Control the speed of the simulation

    # Extract angular velocities for the current step
    omega_x = angular_velocities[i, 0]
    omega_y = angular_velocities[i, 1]
    omega_z = angular_velocities[i, 2]

    # Update the satellite's orientation using the angular velocities
    satellite.rotate(angle=omega_x * dt, axis=vector(1, 0, 0))  # Rotation around x-axis
    satellite.rotate(angle=omega_y * dt, axis=vector(0, 1, 0))  # Rotation around y-axis
    satellite.rotate(angle=omega_z * dt, axis=vector(0, 0, 1))  # Rotation around z-axis

