import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as ani

n = 2000
L = 50
dt = .01
nx = 50    # number of pheromone/grass cells along one direction
ny = 50
p0 = .01    # contribution of a normal locust to pheromone rate
p1 = .05     # contribution of a swarming locust to pheromone rate
pt = .99     # pheromone decay rate
pActivateThresh = 1
pDeactivateThresh = .5
activateProb = .3

x = np.random.rand(n, 2)*L
v = (np.random.rand(n, 2)-.5)*L
p = np.zeros([nx, ny])
active = np.zeros([n])

def step():
    global x, v, p
    x = (x + v*dt)%L

    for i in range(len(x)):     # update pheromones
        xi = int(x[i, 0]*nx/L)
        yi = int(x[i, 1]*ny/L)

        if not active[i]:
            p[xi, yi] += p0
            if (p[xi, yi] > pActivateThresh) and (np.random.rand() < activateProb):
                active[i] = 1
        if (active[i]):
            if (p[xi, yi] < pDeactivateThresh):
                active[i] = 0
            p[xi, yi] += p1
    p *= pt



fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(-.5, L-.5)
plt.ylim(-.5, L-.5)
ax.set_aspect('equal', adjustable='box')

particles = ax.scatter([], [], c=[], s=1, vmin=-.2, vmax=1.2, cmap='RdYlGn', alpha=.5)
background = ax.imshow(np.zeros([nx,nx]), cmap='Reds', vmin=0, vmax=5)

def init():
    global x, v, p
    background.set_array(p)
    particles.set_offsets(x)
    return background, particles

def animate(i):
    global x, v, p
    step()
    particles.set_offsets(x)
    particles.set_array(1-active)
    background.set_array(p)
    return background, particles

animation = ani.FuncAnimation(fig, animate, frames=600,
                              interval=20, blit=True, init_func=init)

plt.show()
