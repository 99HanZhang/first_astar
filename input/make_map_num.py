import json
import matplotlib.pyplot as plt

def make_map(map_path):
    """
    加载地图JSON文件并返回地图元素。
    """
    with open(map_path, 'r', encoding='UTF-8') as f:
        map_data = json.load(f)
    ox = map_data['obstacles']['ox']
    oy = map_data['obstacles']['oy']
    sp = map_data['start_position']
    gp = map_data['parking_sport']
    map_size = map_data['map_size']
    return ox, oy, sp, gp, map_size

def get_obs(map_path_dyobs):
    """
    加载动态障碍物JSON文件。
    """
    with open(map_path_dyobs, 'r', encoding='UTF-8') as f:
        map_dyobs = json.load(f)
    dyobs_x = map_dyobs['X']
    dyobs_y = map_dyobs['Y']
    dyobs_yaw = map_dyobs['Yaw']
    return dyobs_x, dyobs_y, dyobs_yaw

def visualize_map_with_parking_spots(map_path):
    """
    可视化地图，并在停车位上显示编号。
    """
    ox, oy, sp, gp, map_size_str = make_map(map_path)

    # 解析地图尺寸
    map_width, map_height = map(int, map_size_str.split('*'))

    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title(f"Map Visualization: {map_path}")

    # 绘制障碍物
    if ox and oy:
        # 假设障碍物是连续的线段，或者点集
        plt.plot(ox, oy, ".k", markersize=3, label="Obstacles")

    # 绘制起始位置
    plt.plot(sp['x'], sp['y'], "Dg", markersize=10, label="Start Position")
    plt.text(sp['x'], sp['y'] + 50, "Start", fontsize=9, color='green')

    # 绘制停车位并标注编号
    for spot_id, spot_info in gp.items():
        pos = spot_info['pos']
        # 停车位的pos字段通常包含四个角的坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
        # 为了绘制矩形和找到中心点，我们需要重新组织这些坐标
        xs = [pos[i] for i in range(0, len(pos), 2)]
        ys = [pos[i] for i in range(1, len(pos), 2)]

        # 闭合矩形以便绘制
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys, "-b", alpha=0.7) # 绘制停车位边界

        # 计算停车位的中心点以放置编号
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        plt.text(center_x, center_y, str(spot_id), color='red', fontsize=8, ha='center', va='center')

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.axis('equal') # 保证x和y轴的比例一致，避免图形变形

    # 设置坐标轴范围，稍微超出地图边界以提供更好的视图
    plt.xlim(-100, map_width + 100)
    plt.ylim(-100, map_height + 100)

    plt.show()

if __name__ == "__main__":
    while True:
        map_file = input("请输入要加载的JSON地图文件路径 (例如: A.json, 或输入 'exit' 退出): ")
        if map_file.lower() == 'exit':
            break
        try:
            visualize_map_with_parking_spots(map_file)
        except FileNotFoundError:
            print(f"错误: 文件 '{map_file}' 未找到。请检查文件路径。")
        except json.JSONDecodeError:
            print(f"错误: 文件 '{map_file}' 不是一个有效的JSON文件。")
        except KeyError as e:
            print(f"错误: JSON文件缺少必要的键值。可能缺少 '{e}'。请检查地图文件格式。")
        except Exception as e:
            print(f"发生未知错误: {e}")