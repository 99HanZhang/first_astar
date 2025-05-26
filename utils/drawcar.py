

import math
import numpy as np
from input import make_car
import utils.draw as draw

C=make_car.C
def draw_car(ax, x, y, yaw, steer, color='red'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB], [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])
    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR], [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)
    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    ax.plot(car[0, :], car[1, :], color)
    ax.plot(frWheel[0, :], frWheel[1, :], color)
    ax.plot(rrWheel[0, :], rrWheel[1, :], color)
    ax.plot(flWheel[0, :], flWheel[1, :], color)
    ax.plot(rlWheel[0, :], rlWheel[1, :], color)

    draw.Arrow(ax,x, y, yaw, C.WB * 0.8, color)  # 修改为在指定的ax上绘制箭头



# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import utils.draw as draw
# from input import make_car

# C=make_car.C

# def draw_car(x, y, yaw, steer, color='red'):
#     car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
#                     [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

#     wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
#                       [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

#     rlWheel = wheel.copy()
#     rrWheel = wheel.copy()
#     frWheel = wheel.copy()
#     flWheel = wheel.copy()

#     Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
#                      [math.sin(yaw), math.cos(yaw)]])

#     Rot2 = np.array([[math.cos(steer), math.sin(steer)],
#                      [-math.sin(steer), math.cos(steer)]])

#     frWheel = np.dot(Rot2, frWheel)
#     flWheel = np.dot(Rot2, flWheel)

#     frWheel += np.array([[C.WB], [-C.WD / 2]])
#     flWheel += np.array([[C.WB], [C.WD / 2]])
#     rrWheel[1, :] -= C.WD / 2
#     rlWheel[1, :] += C.WD / 2

#     frWheel = np.dot(Rot1, frWheel)
#     flWheel = np.dot(Rot1, flWheel)

#     rrWheel = np.dot(Rot1, rrWheel)
#     rlWheel = np.dot(Rot1, rlWheel)
#     car = np.dot(Rot1, car)

#     frWheel += np.array([[x], [y]])
#     flWheel += np.array([[x], [y]])
#     rrWheel += np.array([[x], [y]])
#     rlWheel += np.array([[x], [y]])
#     car += np.array([[x], [y]])

#     plt.plot(car[0, :], car[1, :], color)
#     plt.plot(frWheel[0, :], frWheel[1, :], color)
#     plt.plot(rrWheel[0, :], rrWheel[1, :], color)
#     plt.plot(flWheel[0, :], flWheel[1, :], color)
#     plt.plot(rlWheel[0, :], rlWheel[1, :], color)
#     draw.Arrow(x, y, yaw, C.WB * 0.8, color)