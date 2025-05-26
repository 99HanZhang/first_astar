
import math
import numpy as np

class C:  # Parameter config
    PI = math.pi

    XY_RESO = 1000  # [m]
    YAW_RESO = np.deg2rad(5.0)  # [rad]
    MOVE_STEP = 20  # [m] path interporate resolution
    N_STEER = 20.0  # steer command number
    COLLISION_CHECK_STEP = 5  # skip number for collision check
    EXTEND_BOUND = 20  # collision check range extended

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    H_COST = 15.0  # Heuristic cost penalty cost

    RF = 450  # [m] distance from rear to vehicle front end of vehicle
    RB = 100  # [m] distance from rear to vehicle back end of vehicle
    W = 300  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 350  # [m] Wheel base
    TR = 50 # [m] Tyre radius
    TW = 100  # [m] Tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle