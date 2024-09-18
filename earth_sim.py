from vpython import sphere, vector, color, rate, curve
import csv

# Initialize lists to store the data from the CSV
time_steps = []
x_pos = []
y_pos = []
z_pos = []

# Read the data from the CSV file
with open('satellite_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        time_steps.append(float(row[0]))
        x_pos.append(float(row[1]))
        y_pos.append(float(row[2]))
        z_pos.append(float(row[3]))

# Constants for visualization
R_earth = 6371e3  # Earth's radius in meters

# Create Earth as a blue sphere
earth = sphere(pos=vector(0, 0, 0), radius=R_earth, color=color.blue, opacity=0.5)

# Create satellite as a red sphere
satellite = sphere(pos=vector(x_pos[0], y_pos[0], z_pos[0]), radius=1e6, color=color.red, make_trail=True, trail_type="curve")

# Create the orbit path
orbit_path = curve(color=color.green)

# Animation loop
for i in range(len(time_steps)):
    rate(100)  # Adjust the speed of the animation
    # Update satellite's position
    satellite.pos = vector(x_pos[i], y_pos[i], z_pos[i])
    # Append position to the orbit path
    orbit_path.append(pos=vector(x_pos[i], y_pos[i], z_pos[i]))
