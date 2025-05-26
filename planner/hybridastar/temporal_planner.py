import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
from planner.hybridastar import astar as astar
from planner.hybridastar import planer_reeds_shepp as rs
from input import make_car

C = make_car.C


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind, time=0):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind
        self.time = time  # 添加时间维度


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree,
                 dyobs_x=None, dyobs_y=None, dyobs_yaw=None):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree
        # 添加动态障碍物信息
        self.dyobs_x = dyobs_x if dyobs_x is not None else []
        self.dyobs_y = dyobs_y if dyobs_y is not None else []
        self.dyobs_yaw = dyobs_yaw if dyobs_yaw is not None else []
        self.time_step = 0.1  # 时间步长为0.1s
        self.max_time = len(self.dyobs_x) * self.time_step if self.dyobs_x else 100.0  # 最大搜索时间


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority


class Path:
    def __init__(self, x, y, yaw, direction, cost=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


def calc_parameters(ox, oy, xyreso, yawreso, kdtree, dyobs_x=None, dyobs_y=None, dyobs_yaw=None):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)
    xw, yw = maxx - minx, maxy - miny
    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree,
                dyobs_x, dyobs_y, dyobs_yaw)


def calc_motion_set():
    s = np.arange(C.MAX_STEER / C.N_STEER,
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)
    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer
    return steer, direc


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


def calc_hybrid_cost(node, hmap, P):
    """改进的代价函数，包含时间偏好"""
    spatial_cost = node.cost + C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    # 添加时间惩罚，鼓励更快到达目标
    time_penalty = node.time * 0.07  # 时间惩罚系数

    # 如果时间超过动态障碍物周期，增加额外惩罚
    if P.dyobs_x and node.time > P.max_time:
        time_penalty += (node.time - P.max_time) * 10

    return spatial_cost + time_penalty


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0
    # 更新时间信息
    move_time = len(path.x) * P.time_step
    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind, n_curr.time + move_time)

    return True, fpath


def calc_rs_path_cost(rspath):
    cost = 0.0
    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST
    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)
    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER
    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])
    return cost


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def get_dyobs_position_at_time(P, t):
    """获取指定时间的动态障碍物位置"""
    if not P.dyobs_x or len(P.dyobs_x) == 0:
        return [], [], []

    # 动态障碍物轨迹循环播放
    cycle_length = len(P.dyobs_x)
    # 确保时间索引正确计算
    time_idx = int(t / P.time_step) % cycle_length

    return P.dyobs_x[time_idx], P.dyobs_y[time_idx], P.dyobs_yaw[time_idx]


def get_vehicle_corners(x, y, yaw):
    """获取车辆四个角点的坐标"""
    # 车辆参数
    length = C.RF + C.RB  # 车长
    width = C.W  # 车宽

    # 车辆中心到后轴的距离
    rear_to_center = C.RB

    # 后轴中心作为参考点
    rear_x = x - rear_to_center * math.cos(yaw)
    rear_y = y - rear_to_center * math.sin(yaw)

    # 计算四个角点相对于后轴中心的位置
    corners = [
        [-C.RB, -width / 2],  # 后左
        [-C.RB, width / 2],  # 后右
        [C.RF, width / 2],  # 前右
        [C.RF, -width / 2]  # 前左
    ]

    # 转换到全局坐标系
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    global_corners = []
    for local_x, local_y in corners:
        global_x = rear_x + local_x * cos_yaw - local_y * sin_yaw
        global_y = rear_y + local_x * sin_yaw + local_y * cos_yaw
        global_corners.append([global_x, global_y])

    return global_corners


def check_vehicle_collision(car1_x, car1_y, car1_yaw, car2_x, car2_y, car2_yaw, safety_margin=75):
    """检查两辆车是否碰撞（动态调整安全边距）"""
    # 计算两车中心距离
    dist = math.hypot(car1_x - car2_x, car1_y - car2_y)

    # 基础安全距离：车辆对角线长度
    car_diagonal = math.hypot(C.RF + C.RB, C.W)
    safe_distance = car_diagonal + safety_margin

    return dist < safe_distance


def is_collision(x, y, yaw, P, current_time=0, adaptive_margin=True):
    """改进的碰撞检测函数，支持自适应安全边距"""
    # 首先检查与静态障碍物的碰撞
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 20  # 碰撞检测扩展半径
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d
        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)
        ids = P.kdtree.query_ball_point([cx, cy], r)

        if ids:
            for i in ids:
                xo = P.ox[i] - cx
                yo = P.oy[i] - cy
                dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                if abs(dx) < r and abs(dy) < C.W / 2 + d:
                    return True

    # 检查与动态障碍物的碰撞
    if P.dyobs_x and len(P.dyobs_x) > 0:
        # 动态调整安全边距
        if adaptive_margin:
            # 根据时间动态调整安全边距
            base_margin = 50  # 基础安全边距
            time_factor = min(current_time / 10.0, 2.0)  # 时间越长，边距越小
            safety_margin = base_margin + 50 / (1 + time_factor)
        else:
            safety_margin = 100  # 固定安全边距

        # 为路径上的每个点检查碰撞
        time_per_step = P.time_step

        for i, (ix, iy, iyaw) in enumerate(zip(x, y, yaw)):
            # 计算当前点对应的时间
            point_time = current_time + i * time_per_step

            # 获取该时刻动态障碍物的位置
            dyobs_x_t, dyobs_y_t, dyobs_yaw_t = get_dyobs_position_at_time(P, point_time)

            # 检查与每个动态障碍物的碰撞
            if isinstance(dyobs_x_t, list):
                # 多个动态障碍物
                for dx, dy, dyaw in zip(dyobs_x_t, dyobs_y_t, dyobs_yaw_t):
                    if check_vehicle_collision(ix, iy, iyaw, dx, dy, dyaw, safety_margin):
                        return True
            else:
                # 单个动态障碍物
                if check_vehicle_collision(ix, iy, iyaw, dyobs_x_t, dyobs_y_t, dyobs_yaw_t, safety_margin):
                    return True

    return False


def is_index_ok(xind, yind, xlist, ylist, yawlist, P, current_time=0):
    """检查节点是否可行"""
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    # 时间约束：如果时间过长，拒绝该节点
    if P.dyobs_x and current_time > P.max_time * 3:  # 允许3个周期的搜索时间
        return False

    # 采样检查碰撞
    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P, current_time):
        return False

    return True


def analystic_expantion(node, ngoal, P):
    """分析性扩展，使用Reed-Shepp路径"""
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    while not pq.empty():
        path = pq.get()
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        # 检查整个路径的碰撞，考虑时间，使用自适应安全边距
        if not is_collision(pathx, pathy, pathyaw, P, node.time, adaptive_margin=True):
            return path

    return None


def calc_next_node(n_curr, c_id, u, d, P):
    """计算下一个节点"""
    step = C.XY_RESO * 2

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    # 计算该动作所需的时间
    move_time = len(xlist) * P.time_step
    next_time = n_curr.time + move_time

    # 检查路径是否可行（包括时间维度的碰撞检测）
    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P, n_curr.time):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalty
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id, next_time)

    return node


def extract_path(closed, ngoal, nstart):
    """从闭集中提取路径"""
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso, dyobs_x=None, dyobs_y=None,
                          dyobs_yaw=None):
    """
    优化的时空混合A*路径规划
    """
    print(f"开始规划，起点: ({sx}, {sy}, {math.degrees(syaw):.1f}°)")
    print(f"目标点: ({gx}, {gy}, {math.degrees(gyaw):.1f}°)")

    if dyobs_x and len(dyobs_x) > 0:
        print(f"动态障碍物数据: {len(dyobs_x)} 个时间点")
        cycle_time = len(dyobs_x) * 0.1
        print(f"障碍物周期: {cycle_time:.1f}s")
    else:
        print("无动态障碍物数据")

    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1, 0.0)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree, dyobs_x, dyobs_y, dyobs_yaw)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    iteration = 0
    max_iterations = 8000  # 最大迭代次数限制

    while True:
        if not open_set:
            print("无法找到路径到达目标点")
            return None

        if iteration >= max_iterations:
            print(f"达到最大迭代次数 {max_iterations}，搜索终止")
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        iteration += 1
        if iteration % 200 == 0:
            print(f"搜索迭代: {iteration}, 当前时间: {n_curr.time:.2f}s, 开放集大小: {len(open_set)}")

        # 检查是否可以直接到达目标
        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            print(f"找到路径! 总迭代次数: {iteration}, 路径时间: {fpath.time:.2f}s")
            fnode = fpath
            break

        # 扩展节点
        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    return extract_path(closed_set, fnode, nstart)


# 用于测试的主函数
def main():
    # 测试场景设置
    sx = 10.0  # 起点x坐标
    sy = 10.0  # 起点y坐标
    syaw = np.deg2rad(0.0)  # 起点航向角
    gx = 50.0  # 终点x坐标
    gy = 50.0  # 终点y坐标
    gyaw = np.deg2rad(90.0)  # 终点航向角

    # 静态障碍物
    ox, oy = [], []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    # 动态障碍物（示例：一个沿直线移动的障碍物）
    dyobs_x = [30.0 + i * 0.5 for i in range(40)]
    dyobs_y = [30.0 for _ in range(40)]
    dyobs_yaw = [np.deg2rad(0.0) for _ in range(40)]

    # 规划参数
    xyreso = 2.0
    yawreso = np.deg2rad(5.0)

    # 调用规划函数
    path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso, dyobs_x, dyobs_y, dyobs_yaw)

    if path:
        print("找到路径!")
        plt.figure(figsize=(10, 10))
        plt.plot(ox, oy, ".k")
        plt.plot(path.x, path.y, "-r", label="Hybrid A* path")
        plt.plot(sx, sy, "og", label="Start")
        plt.plot(gx, gy, "xb", label="Goal")

        # 绘制动态障碍物的轨迹
        plt.plot(dyobs_x, dyobs_y, "--g", label="Dynamic obstacle")

        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.show()
    else:
        print("未找到路径!")


if __name__ == "__main__":
    main()