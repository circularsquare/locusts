import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as ani

ni = 200   # initial number of locusts
n = 5000
L = 50
dt = .01
nx = 50    # number of pheromone/grass cells along one direction
ny = 50
nBig = 10   # number of big cells along one direction
p0 = .01    # normal locust's pheromone emission
p1 = .05    # swarming locust's pheromone emission
pt = .02    # pheromone decay rate
f0 = .002    # amount a normal locust eats
f1 = .005    # amount a swarming locust eats
ft = .005    # food growth rate
birth0 = .003    # birth rate (if fed)
birth1 = .010    
death0 = .2    # death rate (if hungry)
death1 = .2
pActivateThresh = 1
pDeactivateThresh = .5
activateProb = .3

x = np.zeros([n, 2]) - 1000          # position
x[:ni] = np.random.rand(ni, 2)*L
v = np.zeros([n, 2])              # velocity
v[:ni] = (np.random.rand(ni, 2)-.5)*10
p = np.zeros([nx, ny])              # pheromones
f = np.ones([nx, ny])              # food
dens = np.zeros([nx, ny])           # number of locusts in each grid square
state = np.zeros([n])              # whether each locust is normal, swarming, or dead (1, 2, 0)
state[:ni] = np.ones([ni])
bigGrid[   

def step():
    global x, v, p, f, state, dens, distances
    x = (x + v*dt)%L

    xgrid = (((x+.5)*nx/L).astype(int))%L

    use bigGrid
    fff = 0
    # find neighbors
    for i in range(n):
        for j in range(n):
            fff += j
    print(fff)

    randoms = np.random.rand(n)
    randoms2 = np.random.rand(n)

    for i in range(n): 
        # xi = (int((x[i, 0]+.5)*nx/L))%L      # locust's grid position
        # yi = (int((x[i, 1]+.5)*ny/L))%L
        xi = xgrid[i, 0]
        yi = xgrid[i, 1]

        if state[i] == 1:           # update pheromones and food
            if (p[xi, yi] > pActivateThresh) and (randoms2[i] < activateProb):
                state[i] = 2
            if f[xi, yi] > f0:
                f[xi, yi] -= f0
                if randoms[i] < birth0:
                    birth(x[i], v[i])
                p[xi, yi] += p0
            elif randoms[i] < death0:
                state[i] = 0  
                x[i] = [-1000,-1000]
                v[i] = [0,0]

        if state[i] == 2:
            if (p[xi, yi] < pDeactivateThresh):
                state[i] = 1
            if f[xi, yi] > f1:
                f[xi, yi] -= f1
                if randoms[i] < birth1:
                    birth(x[i], v[i])
                p[xi, yi] += p1
            elif randoms[i] < death1:
                state[i] = 0
                x[i] = [-1000,-1000]
                v[i] = [0,0]


    # update velocities
    # go towards food + pheromones
    # go with neighbors

    p *= (1-pt)     # decay pheromones
    p = .96*f + .01*(np.roll(p, 1, axis=0) + 
                np.roll(p, 1, axis=1) +      # diffuse food
                np.roll(p, -1, axis=0) + 
                np.roll(p, -1, axis=-1))   

    f = f*(1 + (1-f)*ft + .001)    # grow food (growth rate peaks at ft, when f=.5)
    f = .96*f + .01*(np.roll(f, 1, axis=0) + 
                    np.roll(f, 1, axis=1) +      # diffuse food
                    np.roll(f, -1, axis=0) + 
                    np.roll(f, -1, axis=-1))   

    print(np.bincount(state.astype(int)))

def birth(pos, vel):
    global x, v, state
    emptySpots = min(list(np.where(state == 0)))
    if len(emptySpots)>0:
        i = emptySpots[0]
        x[i] = pos
        v[i] = vel + (.5-np.random.rand(2))*2
        state[i] = 1 
    else:
        print('max population reached')
    



fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(-.5, L-.5)
plt.ylim(-.5, L-.5)
ax.set_aspect('equal', adjustable='box')

particles = ax.scatter([], [], c=[], s=2, vmin=.5, vmax=2.5, cmap='YlOrBr', alpha=.5)
background = ax.imshow(np.zeros([nx,nx]), cmap='Greens', vmin=0, vmax=1.5)

def init():
    global x, v, p
    background.set_array(p)
    particles.set_offsets(x)
    return background, particles

def animate(i):
    global x, v, p
    step()
    particles.set_offsets(x[:, [1, 0]])
    particles.set_array(state)
    background.set_array(f)
    return background, particles

animation = ani.FuncAnimation(fig, animate, frames=600,
                              interval=20, blit=True, init_func=init)

plt.show()
