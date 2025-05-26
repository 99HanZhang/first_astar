import json
import math
import matplotlib.pyplot as plt
import input.make_map as mp
from utils import drawcar as tools, reeds_shepp as rs
from input import make_car
import numpy as np
import imageio

def replay(map_scene,output_result,dynamic_obs):
    picture_scene =  map_scene.replace('/', '/').replace('.json', '.jpg')
    C = make_car.C
    ox, oy,sp,gp = mp.make_map(map_scene)
    gx, gy, gyaw0 = gp['3']['x_end'], gp['3']['y_end'], gp['3']['yaw']
    with open(output_result,'r',encoding='UTF-8') as f:
        result=json.load(f)
    x = result['output_x']
    y = result['output_y']
    yaw = result['output_yaw']
    direction = result['output_dir']

    with open(dynamic_obs,'r',encoding='UTF-8') as f:
        result=json.load(f)
    dox = result['X']
    doy = result['Y']
    doyaw = result['Yaw']


    plt.rcParams['xtick.direction'] = 'in'
    picture = plt.imread(picture_scene)
    ox1,ox2,oy1,oy2 = min(ox), max(ox), min(oy), max(oy)
    frames = []  # 用于保存每一帧图像
    fig, ax = plt.subplots()
    i=0
    ax.imshow(picture, extent=[ox1, ox2, oy1, oy2], aspect='auto')
    ax.set_title("Simulation Result", loc='left', fontweight="heavy")
    ax.axis("equal")
    for k in range(len(x)):
        if k % 5 == 0:
            ax.lines.clear()
            ax.plot(x, y, linewidth=0.5, color='b', linestyle='--')
            ax.plot(dox, doy, linewidth=0.5, color='r', linestyle='--')
            # ax.plot(ox, oy, ",k")
            if k < len(x) - 2:
                dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
                steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))
            else:
                steer = 0.0
            tools.draw_car(ax, gx, gy, gyaw0, 0.0, 'dimgray')
            tools.draw_car(ax, x[k], y[k], yaw[k], steer)
            if i >= len(dox):
                i = 0
            tools.draw_car(ax, dox[i], doy[i], doyaw[i], steer)
            i = i+5
            plt.pause(0.2)
            # plt.show()
    # 将所有保存的帧图像整合成 GIF 动态图
    # imageio.mimsave(gif_scene, frames, fps=15)
    # if output_result[-6].isdigit():
    #     if Num[-2].isdigit():
    #         print(f"车位{Num[-2:]}仿真完成")
    #     else:
    #         print(f"车位{Num[-1]}仿真完成")
    # else:
    #     print("仿真结束!")
    plt.close()