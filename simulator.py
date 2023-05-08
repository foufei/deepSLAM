import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self):
        # Set up the physics simulation
        p.connect(p.GUI) # or p.DIRECT for headless mode
        p.setGravity(0, 0, -9.81) # set gravity in z direction
        p.setTimeStep(1./240.) # set simulation time step
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # add data path for built-in assets

        # TODO: load the agent model
        self.agent = p.loadURDF("humanoid/humanoid.urdf", [0,0,0.9], p.getQuaternionFromEuler([np.pi/2,0,0]), globalScaling=0.25)
        self.obstacles = []
        self.dynamics = []
        self.initialize_env()

        self.pixelWidth = 320
        self.pixelHeight = 200

        self.camTargetPos = [0, 0, 0]
        self.cameraUp = [0, 0, 1]
        self.cameraPos = [1, 1, 1]

        self.pitch = -10.0
        self.roll = 0

        self.upAxisIndex = 2
        self.camDistance = 4
        self.nearPlane = 0.01
        self.farPlane = 100

        self.fov = 60
        self.projectionMatrix = None

    def initialize_env(self, num_obstacles=2, num_dynamics=4, dynamic_velocity=50, distance=20):
        # Create a plaza with random static obstacles
        plazaId = p.loadURDF("plane.urdf", [0, 0, 0])

        # Simulate the static cubes in the plaza as the obstacles
        for _ in range(num_obstacles):
            x = np.random.uniform(low=-distance, high=distance)
            y = np.random.uniform(low=-distance, high=distance)
            obstacle = p.loadURDF("cube.urdf", [x, y, 1], globalScaling=2)

            self.obstacles.append(obstacle)

        # Simulate the crowd of dynamic cubes moving in the plaza
        for _ in range(num_dynamics):
            # Set the initial position
            x = np.random.uniform(low=-distance, high=distance)
            y = np.random.uniform(low=-distance, high=distance)
            dynamic = p.loadURDF("sphere2.urdf", [x, y, 1], p.getQuaternionFromEuler([0,0,0]), globalScaling=1.5)

            # Set the velocity
            v_x = np.random.uniform(-dynamic_velocity, dynamic_velocity)
            v_y = np.random.uniform(-dynamic_velocity, dynamic_velocity)
            p.resetBaseVelocity(dynamic, [v_x, v_y, 0], [0, 0, 0])

            self.dynamics.append(dynamic)

    def initialize_camera(self):
        self.aspect = self.pixelWidth / self.pixelHeight

        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearPlane, self.farPlane)


    def get_image(self, yaw):
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, yaw, self.pitch, self.roll, self.upAxisIndex)
        img_arr = p.getCameraImage( self.pixelWidth,
                                    self.pixelHeight,
                                    viewMatrix,
                                    self.projectionMatrix,
                                    shadow=1,
                                    lightDirection=[1, 1, 1],
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL )
        return img_arr

    def run(self, num_timesteps = 2400):
        while(1):
            img = np.random.rand(200, 320)
            image = plt.imshow(img, interpolation='none', animated=True, label="blah")
            for yaw in range(0, 360, 10):
                p.stepSimulation() # simulate one time step

                img_arr = self.get_image(yaw)
                w = img_arr[0]  # width of the image, in pixels
                h = img_arr[1]  # height of the image, in pixels
                rgb = img_arr[2]  # color data RGB
                dep = img_arr[3]  # depth data

                np_img_arr = np.reshape(rgb, (h, w, 4))
                np_img_arr = np_img_arr * (1. / 255.)
                image.set_data(np_img_arr)
                plt.pause(0.01)

                time.sleep(1./240.) # wait for a short time


