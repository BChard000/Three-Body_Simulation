import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Mass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.trail_x = [x]
        self.trail_y = [y]

    def set_xy(self, x, y):
        self.x = x
        self.y = y
        self.trail_x.append(x)
        self.trail_y.append(y)

class ThreeBody:
    def __init__(self):
        self.n = 3  # number of interacting bodies
        # state = [x1, vx1, y1, vy1, x2, vx2, y2, vy2, x3, vx3, y3, vy3, t]
        self.state = np.zeros(4 * self.n + 1)
        self.state[0], self.state[2] = 0, 0  # initial position of mass1
        self.state[4], self.state[6] = 1, 0  # initial position of mass2
        self.state[8], self.state[10] = 0, 1  # initial position of mass3
        self.mass1 = Mass(self.state[0], self.state[2])
        self.mass2 = Mass(self.state[4], self.state[6])
        self.mass3 = Mass(self.state[8], self.state[10])

    def compute_force(self, state):
        force = np.zeros(2 * self.n)
        G = 1  # Gravitational constant
        masses = [1, 1, 1]  # Masses of the three bodies
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dx = state[4 * j] - state[4 * i]
                    dy = state[4 * j + 2] - state[4 * i + 2]
                    r = np.sqrt(dx**2 + dy**2)
                    if r == 0:  # avoid division by zero
                        continue
                    f = G * masses[i] * masses[j] / r**2
                    theta = np.arctan2(dy, dx)
                    force[2 * i] += f * np.cos(theta)
                    force[2 * i + 1] += f * np.sin(theta)
        return force

    def get_rate(self, state):
        force = self.compute_force(state)
        rate = np.zeros_like(state)
        for i in range(self.n):
            i4 = 4 * i
            rate[i4] = state[i4 + 1]  # x rate is vx
            rate[i4 + 1] = force[2 * i]  # vx rate is fx
            rate[i4 + 2] = state[i4 + 3]  # y rate is vy
            rate[i4 + 3] = force[2 * i + 1]  # vy rate is fy
        rate[-1] = 1  # time rate is last
        return rate

    def step(self, dt):
        k1 = self.get_rate(self.state)
        k2 = self.get_rate(self.state + 0.5 * dt * k1)
        k3 = self.get_rate(self.state + 0.5 * dt * k2)
        k4 = self.get_rate(self.state + dt * k3)
        self.state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.mass1.set_xy(self.state[0], self.state[2])
        self.mass2.set_xy(self.state[4], self.state[6])
        self.mass3.set_xy(self.state[8], self.state[10])

# Initialize the simulation
three_body = ThreeBody()

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

mass1_plot, = ax.plot([], [], 'ro')
mass2_plot, = ax.plot([], [], 'go')
mass3_plot, = ax.plot([], [], 'bo')
trail1_plot, = ax.plot([], [], 'r--', linewidth=0.5)
trail2_plot, = ax.plot([], [], 'g--', linewidth=0.5)
trail3_plot, = ax.plot([], [], 'b--', linewidth=0.5)

def init():
    mass1_plot.set_data([], [])
    mass2_plot.set_data([], [])
    mass3_plot.set_data([], [])
    trail1_plot.set_data([], [])
    trail2_plot.set_data([], [])
    trail3_plot.set_data([], [])
    return mass1_plot, mass2_plot, mass3_plot, trail1_plot, trail2_plot, trail3_plot

def update(frame):
    dt = 0.01
    three_body.step(dt)
    mass1_plot.set_data([three_body.mass1.x], [three_body.mass1.y])
    mass2_plot.set_data([three_body.mass2.x], [three_body.mass2.y])
    mass3_plot.set_data([three_body.mass3.x], [three_body.mass3.y])
    trail1_plot.set_data(three_body.mass1.trail_x, three_body.mass1.trail_y)
    trail2_plot.set_data(three_body.mass2.trail_x, three_body.mass2.trail_y)
    trail3_plot.set_data(three_body.mass3.trail_x, three_body.mass3.trail_y)
    return mass1_plot, mass2_plot, mass3_plot, trail1_plot, trail2_plot, trail3_plot

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True)
plt.show()