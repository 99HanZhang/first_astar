import json
import input.make_map as mp
from planner.hybridastar import planner as planner
from input import make_car
from utils import replay
from utils import map_display as mdp
import matplotlib.pyplot as plt

def main():

    # 输入input文件夹下场景文件
    map_path = '../input/B03.json' #读取静态地图
    map_path_dyobs = '../input/B03_dyobs.json'  #读取动态地图
    # mdp.map_display(map_path) #  仅绘制地图

    ox, oy,sp,gp = mp.make_map(map_path)
    sx, sy, syaw0 = sp['x'], sp['y'], sp['yaw']
    C = make_car.C

    # 获取动态障碍物
    dyobs_x,dyobs_y,dyobs_yaw=mp.get_obs(map_path_dyobs)

    # 获取目标停车位
    park = '7'
    gx, gy, gyaw0 = gp[park]['x_end'], gp[park]['y_end'], gp[park]['yaw']

    # 规划算法
    path = planner.hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0, ox, oy, C.XY_RESO, C.YAW_RESO,dyobs_x,dyobs_y,dyobs_yaw)
    # 算法测试结果保存
    if not path:
        print("Searching failed!")
        return
    output_dit={
        "parking":park,
        "output_x":path.x,
        "output_y": path.y,
        "output_yaw": path.yaw,
        "output_dir": path.direction,
    }
    with open(f"../output/result_{map_path.split('/')[-1].split('.json')[0]}_{park}.json", "w") as file:
        json.dump(output_dit, file)

    # 仿真回放
    result_path = f"../output/result_{map_path.split('/')[-1].split('.json')[0]}_{park}.json"
    replay.replay(map_path, result_path,map_path_dyobs)
if __name__ == '__main__':
    main()