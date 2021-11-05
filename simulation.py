import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as ani

ni = 200   # initial number of locusts
n = 5000
L = 60
t = 0
dt = .05
nx = L   # number of pheromone/grass cells along one direction
ny = L
nBig = 25   # number of big cells along one direction
p0 = 1    # normal locust's pheromone emission
p1 = 5    # swarming locust's pheromone emission
pt = 2    # pheromone decay rate
f0 = .2    # amount a normal locust eats
f1 = .5    # amount a swarming locust eats
ft = .2    # food growth rate
birth0 = .1    # birth rate (if fed)
birth1 = .5    
death0 = 1    # death rate (if hungry)
death1 = 2
deathLonely = .1
pActivateThresh = 1
pDeactivateThresh = .5
activateProb = .3
followNeighbors = .1
followFood = .2
followPheromones = .02
maxSpeed0 = 5
maxSpeed1 = 10

x = np.zeros([n, 2]) - 600          # position
x[:ni] = np.random.rand(ni, 2)*L
v = np.zeros([n, 2])              # velocity
v[:ni] = (np.random.rand(ni, 2)-.5)*10
p = np.zeros([nx, ny])              # pheromones
f = np.zeros([nx, ny])+.3              # food
dens = np.zeros([nx, ny])           # number of locusts in each grid square
state = np.zeros([n])              # whether each locust is normal, swarming, or dead (1, 2, 0)
state[:ni] = np.ones([ni])
bigGrid = np.zeros([nBig, nBig])    # grid used for finding locust interactions
bgPop = np.zeros([nBig, nBig])
bgVelocity = np.zeros([nBig, nBig, 2])


def step():
    global t, x, v, p, f, state, dens, distances
    x = (x + v*dt)%L

    temp = max(-.2, np.cos(t/10)+.5) #temperature

    xgrid = (((x+.5)*nx/L).astype(int))%L
    xbGrid = (((x+L/nBig)*nBig/L).astype(int))%nBig
    randoms = np.random.rand(n)
    randoms2 = np.random.rand(n)
    randoms3 = np.random.rand(n, 2)-.5
    for i in range(n):
        if state[i] > 0:
            bgPop[xbGrid[i, 0], xbGrid[i, 1]] += 1
            bgVelocity[xbGrid[i, 0], xbGrid[i, 1], :] += v[i, :]
    for i in range(n):
        if state[i] > 0:
            xbi = xbGrid[i, 0]
            ybi = xbGrid[i, 1]
            v[i] = v[i]*(1-followNeighbors) + bgVelocity[xbi, ybi, :]/bgPop[xbi, ybi]*followNeighbors

            xi = xgrid[i, 0]
            yi = xgrid[i, 1]
            v[i, 1] += followFood * (f[xi, (yi+1)%L] - f[xi, (yi-1)%L])
            v[i, 1] *= (1 + .3*followFood * (f[xi, (yi+1)%L] + f[xi, (yi+1)%L] - 2*f[xi, yi]))
            v[i, 0] += followFood * (f[(xi+1)%L, yi] - f[(xi-1)%L, yi])
            v[i, 0] *= (1 + .3*followFood * (f[(xi+1)%L, yi] + f[(xi-1)%L, yi] - 2*f[xi, yi]))

            v[i, 1] += followPheromones * (p[xi, (yi+1)%L] - p[xi, (yi-1)%L])
            v[i, 0] += followPheromones * (p[(xi+1)%L, yi] - p[(xi-1)%L, yi])

            if state[i] == 1:
                if v[i, 0]*v[i, 0] + v[i, 1]*v[i, 1] > maxSpeed0*maxSpeed0*(.5+.5*temp):
                    v[i] = v[i]*.9
            else:
                if v[i, 0]*v[i, 0] + v[i, 1]*v[i, 1] > maxSpeed1*maxSpeed1*(.5+.5*temp):
                    v[i] = v[i]*.9

            v[i] += randoms3[i]


    for i in range(n): 
        xi = xgrid[i, 0]
        yi = xgrid[i, 1]

        if state[i] == 1:           # update pheromones and food
            if (p[xi, yi]+temp-.5 > pActivateThresh) and (randoms2[i] < activateProb):
                state[i] = 2
            if f[xi, yi] > f0*dt*(temp+.5):
                f[xi, yi] -= f0*dt*(temp+.5)
                if randoms[i] < birth0*dt*(temp+.5):
                    birth(x[i], v[i])
                p[xi, yi] += p0*dt*(temp+.5)
            elif randoms[i] < death0*dt:
                state[i] = 0  
                x[i] = [-600,-600]
                v[i] = [0,0]
                f[xi, yi] += .1
            if p[xi, yi] < .1 and randoms[i] < deathLonely*dt:
                state[i] = 0  
                x[i] = [-600,-600]
                v[i] = [0,0]
                f[xi, yi] += .1

        if state[i] == 2:
            if (p[xi, yi]+temp-.5 < pDeactivateThresh):
                state[i] = 1
            if f[xi, yi] > f1*dt*(temp+.5):
                f[xi, yi] -= f1*dt*(temp+.5)
                if randoms[i] < birth1*dt*(temp+.5):
                    birth(x[i], v[i])
                p[xi, yi] += p1*dt*(temp+.5)
            elif randoms[i] < death1*dt:
                state[i] = 0
                x[i] = [-600,-600]
                v[i] = [0,0]
                f[xi, yi] += .1


    p *= (1-pt*dt)     # decay pheromones
    p = (1-.4*dt)*p + .1*dt*(np.roll(p, 1, axis=0) + 
                np.roll(p, 1, axis=1) +      # diffuse pheromones
                np.roll(p, -1, axis=0) + 
                np.roll(p, -1, axis=-1))   

    f = f*(1 + (1-f)*ft*dt*temp + .001*dt*temp)    # grow food (growth rate peaks at ft, when f=.5)
    f = (1-.4*dt)*f + .1*dt*(np.roll(f, 1, axis=0) + 
                    np.roll(f, 1, axis=1) +      # diffuse food
                    np.roll(f, -1, axis=0) + 
                    np.roll(f, -1, axis=-1))   
    t += dt

    #print(np.bincount(state.astype(int)))

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
ax2 = fig.add_subplot(122)
ax.set_xlim(.5, L-.5)
ax.set_ylim(.5, L-.5)
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
ax2.set_xlim(.5, L-.5)
ax2.set_ylim(.5, L-.5)
ax2.axis('off')
ax2.set_aspect('equal', adjustable='box')

particles = ax.scatter([], [], c='#ff4d40', s=2, vmin=.5, vmax=2.5, cmap='YlOrBr', alpha=.5)
background = ax.imshow(np.zeros([nx,nx]), cmap='Greens', vmin=0, vmax=1.5)
background2 = ax.imshow(np.zeros([nx,nx]), cmap='Reds', vmin=0, vmax=1.5, alpha=.2)

def init():
    global x, v, p, t
    background.set_array(f)
    background2.set_array(p)
    particles.set_offsets(x)
    return background, background2, particles

def animate(i):
    global x, v, p, t
    step()
    background.set_array(f)
    background2.set_array(p)
    particles.set_offsets(x[:, [1, 0]])
    #particles.set_array(state)

    return background, background2, particles

animation = ani.FuncAnimation(fig, animate, frames=600,
                              interval=20, blit=True, init_func=init)

plt.show()
