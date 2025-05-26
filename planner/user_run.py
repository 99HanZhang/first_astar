import json
import os



import input.make_map as mp
from input import make_car
from utils import replay
from utils import map_display as mdp
def main():
    in_path="../input/" #地图文件
    in_path_dyobs = "../input/" #障碍物文件
    # 遍历input文件夹下所有文件
    for root,dirs,files in os.walk(in_path):
        for file in files:
            if "json" in file: # 找到.json文件
                # mdp.map_display(map_path) #  仅绘制地图
                ox, oy,sp,gp = mp.make_map(in_path+file) #获取地图信息（障碍物，起点，车位）
                dyobs_x, dyobs_y, dyobs_yaw = mp.get_obs(in_path_dyobs)   # 获取动态障碍物
                sx, sy, syaw0 = sp['x'], sp['y'], sp['yaw']# 获取起点的信息
                C = make_car.C # 获取车辆的属性信息
                # 遍历所有车位
                for i in range(1,len(gp)+1):
                    # 获取该车位的信息
                    gx, gy, gyaw0 = gp[str(i)]['x_end'], gp[str(i)]['y_end'], gp[str(i)]['yaw']
                    # 请在此调用planer文件夹下的规控算法
                    try:
                        path = ""
                    except:
                        continue
                   # 算法测试结果保存
                    if not path:
                        print("Searching failed!")
                        continue
                    output_dit={
                        "parking":str(i),
                        "output_x": path.x,
                        "output_y": path.y,
                        "output_yaw": path.yaw,
                        "output_dir": path.direction,
                    }
                    with open(f"../output/result_{str(file).split('.json')[0]}_{str(i)}.json", "w") as file1:
                        json.dump(output_dit, file1)

                    # 仿真回放
                    # result_path = f"../Onsite_Parking/output/result_{map_path.split('/')[-1].split('.json')[0]}_{str(i)}.json"
                    # replay.replay(in_path, result_path,in_path_dyobs)

if __name__ == '__main__':
    main()