"""
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
"""

import asyncore
import base64
import math
import time
from io import BytesIO
from threading import Thread

import numpy as np
from PIL import Image
from donkey_gym.core.fps import FPSTimer
from donkey_gym.core.tcp_server import IMesgHandler, SimServer


class DonkeyUnitySimContoller:
    # cross track error max
    # CTE_MAX_ERR = 3.5

    def __init__(self, level, time_step=0.05, port=9090):
        self.level = level
        self.time_step = time_step
        self.verbose = False
        self.wait_time_for_obs = 0.1

        # sensor size - height, width, depth
        self.camera_img_size = (80, 160, 3)

        self.address = ('0.0.0.0', port)

        self.handler = DonkeyUnitySimHandler(level, time_step=time_step)
        self.server = SimServer(self.address, self.handler)

        self.thread = Thread(target=asyncore.loop)
        self.thread.daemon = True
        self.thread.start()

    def wait_until_loaded(self):
        while not self.handler.loaded:
            print("waiting to load..")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        return self.handler.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    # cross track error max
    CTE_MAX_ERR = 3.5
    FPS = 60.0

    def __init__(self, level, time_step=0.05):
        self.iSceneToLoad = level
        self.time_step = time_step
        self.wait_time_for_obs = 0.1
        self.sock = None
        self.loaded = False
        self.verbose = False
        self.timer = FPSTimer(verbose=1)

        # sensor size - height, width, depth
        self.camera_img_size = (80, 160, 3)
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.last_throttle = 0.0
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.steering_angle  = 0.0
        self.current_step = 0

        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}

    def on_connect(self, socket_handler):
        self.sock = socket_handler

    def on_disconnect(self):
        self.sock = None

    def on_recv_message(self, message):
        if 'msg_type' not in message:
            print('expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('unknown message type', msg_type)

    # ------- Env interface ---------- #

    def reset(self):
        if self.verbose:
            print("reseting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.current_step = 0
        self.send_reset_car()
        time.sleep(1.0)
        self.timer.reset()

    def get_sensor_size(self):
        return self.camera_img_size

    def take_action(self, action):
        if self.verbose:
            print("take_action")

        # Static throttle
        throttle = 0.5
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(action[0], throttle)

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {}

        self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        # Workaround for big error at start.
        if math.fabs(self.cte) > 2 * self.CTE_MAX_ERR and self.current_step < 10:
            print("Too high error, ignoring {:.2f}".format(self.cte))
            return True
        return self.hit != "none" or math.fabs(self.cte) > self.CTE_MAX_ERR

    # ------ RL interface ----------- #

    # Use velocity (m/s) as reward for every step,
    # except when episode done (failed).
    def calc_reward(self, done):
        if done:
            return 0.0

        velocity = self.last_throttle * (1.0 / self.FPS)
        return velocity

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Crop to the zone of interest - remove top third.
        # Crop image to size 80x160x3.
        self.image_array = np.delete(np.asarray(image), np.s_[0:40:], axis=0)

        # name of object we just hit. "none" if nothing.
        if self.hit == "none":
            self.hit = data["hit"]

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.steering_angle = data['steering_angle']

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 3 scenes available now.
        try:
            self.cte = data["cte"]
        except KeyError:
            print("No CTE")
            pass

    def on_scene_selection_ready(self, _data):
        print("SceneSelectionReady ")
        self.send_get_scene_names()

    def on_car_loaded(self, _data):
        if self.verbose:
            print("car loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            if self.verbose:
                print("SceneNames:", names)
            self.send_load_scene(names[self.iSceneToLoad])

    def send_control(self, steer, throttle):
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__(), 'brake': '0.0'}
        self.queue_message(msg)

    def send_reset_car(self):
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_get_scene_names(self):
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def queue_message(self, msg):
        if self.sock is None:
            if self.verbose:
                print('skipping:', msg)
            return

        if self.verbose:
            print('sending', msg)
        self.sock.queue_message(msg)
