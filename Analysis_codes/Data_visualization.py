import matplotlib.pyplot as plt
import matplotlib.animation as animation

positions = [(2.5, 2.5), (4, 4), (1, 4), (1, 1), (4, 1), (2.5, 2.5)]

fig, ax = plt.subplots()
# Initialize at the first position
player_dot, = ax.plot([positions[0][0]], [positions[0][1]], 'ro', markersize=10)
court_lines = [(0, 0), (0, 5), (5, 5), (5, 0), (0, 0)]

ax.plot(*zip(*court_lines), color='gray')
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)

def update(frame):
    player_dot.set_data([positions[frame][0]], [positions[frame][1]])
    return player_dot,

ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=500, blit=True)
ani.save('footwork_demo.gif', writer='pillow')