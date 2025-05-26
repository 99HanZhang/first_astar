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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Onsite_Parking/")

C = make_car.C


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
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


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
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


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0

    def put(self, item, priority):
        self.queue[item] = priority

    def get(self):
        return self.queue.popitem()[0]


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)
    xw, yw = maxx - minx, maxy - miny
    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)


def calc_motion_set():
    """改进的运动集合 - 更实用的泊车动作"""
    # 使用更合理的转向角度组合
    # 减少极端转向，增加中等转向的选择
    steer_angles = [
        -C.MAX_STEER * 0.8,  # 大角度左转
        -C.MAX_STEER * 0.5,  # 中角度左转
        -C.MAX_STEER * 0.2,  # 小角度左转
        0.0,  # 直行
        C.MAX_STEER * 0.2,  # 小角度右转
        C.MAX_STEER * 0.5,  # 中角度右转
        C.MAX_STEER * 0.8  # 大角度右转
    ]

    # 前进和后退
    direc = [1.0 for _ in range(len(steer_angles))] + [-1.0 for _ in range(len(steer_angles))]
    steer = steer_angles + steer_angles

    return steer, direc


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)
    return ind


def calc_hybrid_cost(node, hmap, P, ngoal=None):
    """改进的代价函数 - 分阶段优化"""
    # 计算到目标的距离，判断是否接近目标
    distance_to_goal = float('inf')
    if ngoal and hasattr(node, 'x') and hasattr(ngoal, 'x') and node.x and ngoal.x:
        dx = node.x[-1] - ngoal.x[-1]
        dy = node.y[-1] - ngoal.y[-1]
        distance_to_goal = math.hypot(dx, dy)

    # 基础启发式代价
    heuristic_cost = 0
    try:
        hx = node.xind - P.minx
        hy = node.yind - P.miny
        if 0 <= hx < len(hmap) and 0 <= hy < len(hmap[0]):
            heuristic_cost = C.H_COST * hmap[hx][hy]
        else:
            if distance_to_goal != float('inf'):
                heuristic_cost = C.H_COST * distance_to_goal / 100
    except (IndexError, TypeError):
        if distance_to_goal != float('inf'):
            heuristic_cost = C.H_COST * distance_to_goal / 100

    base_cost = node.cost + heuristic_cost

    # 分阶段的方向变化惩罚
    direction_penalty = 0
    if hasattr(node, 'directions') and len(node.directions) > 1:
        changes = 0
        prev_dir = node.directions[0]
        for curr_dir in node.directions[1:]:
            if curr_dir != prev_dir:
                changes += 1
                prev_dir = curr_dir

        # 根据距离目标的远近调整惩罚力度
        if distance_to_goal > 500:  # 远离目标时，轻微惩罚方向变化
            direction_penalty = changes * 30
        elif distance_to_goal > 200:  # 中等距离时，中等惩罚
            direction_penalty = changes * 60
        else:  # 接近目标时，严厉惩罚方向变化
            direction_penalty = changes * 120

    # 路径效率惩罚
    length_penalty = 0
    if hasattr(node, 'x'):
        length_penalty = len(node.x) * 0.3

    return base_cost + direction_penalty + length_penalty


def is_collision(x, y, yaw, P):
    """改进的碰撞检测 - 修复避障问题"""
    for ix, iy, iyaw in zip(x, y, yaw):
        # 适中的安全边距
        d = 15  # 稍微减小安全边距，避免过于保守
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d

        # 车辆中心点
        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        # 查找附近的障碍物
        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        # 检查每个障碍物
        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy

            # 转换到车辆坐标系
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            # 更精确的碰撞检测
            # 考虑车辆的实际尺寸
            if abs(dx) < (C.RF + C.RB) / 2.0 + d and abs(dy) < C.W / 2 + d:
                return True
    return False


def is_near_goal(node, ngoal, threshold_distance=300, threshold_angle=math.pi / 4):
    """判断是否接近目标 - 决定是否使用RS曲线"""
    if not (hasattr(node, 'x') and hasattr(ngoal, 'x') and node.x and ngoal.x):
        return False

    # 距离检查
    dx = node.x[-1] - ngoal.x[-1]
    dy = node.y[-1] - ngoal.y[-1]
    distance = math.hypot(dx, dy)

    # 角度检查
    angle_diff = float('inf')
    if hasattr(node, 'yaw') and hasattr(ngoal, 'yaw') and node.yaw and ngoal.yaw:
        angle_diff = abs(rs.pi_2_pi(node.yaw[-1] - ngoal.yaw[-1]))

    return distance < threshold_distance and angle_diff < threshold_angle


def generate_simple_path(n_curr, ngoal, step_size=C.MOVE_STEP):
    """生成简单直线或圆弧路径 - 替代中远距离的RS曲线"""
    sx, sy, syaw = n_curr.x[-1], n_curr.y[-1], n_curr.yaw[-1]
    gx, gy = ngoal.x[-1], ngoal.y[-1]

    # 计算到目标的方向
    target_angle = math.atan2(gy - sy, gx - sx)
    angle_diff = rs.pi_2_pi(target_angle - syaw)

    # 如果角度差异较小，使用直线路径
    if abs(angle_diff) < math.pi / 6:
        return generate_straight_path(sx, sy, syaw, gx, gy, step_size)
    else:
        return generate_arc_path(sx, sy, syaw, target_angle, step_size)


def generate_straight_path(sx, sy, syaw, gx, gy, step_size):
    """生成直线路径"""
    distance = math.hypot(gx - sx, gy - sy)
    steps = max(1, int(distance / step_size))

    x_list = [sx + (gx - sx) * i / steps for i in range(steps + 1)]
    y_list = [sy + (gy - sy) * i / steps for i in range(steps + 1)]
    yaw_list = [syaw] * (steps + 1)
    direction_list = [1] * (steps + 1)

    # 创建简单的路径对象
    class SimplePath:
        def __init__(self, x, y, yaw, directions):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.directions = directions

    return SimplePath(x_list, y_list, yaw_list, direction_list)


def generate_arc_path(sx, sy, syaw, target_angle, step_size):
    """生成圆弧路径 - 平滑转向"""
    # 简单的圆弧路径生成
    angle_diff = rs.pi_2_pi(target_angle - syaw)

    # 使用中等转向半径
    turn_radius = C.WB / math.tan(C.MAX_STEER * 0.5)
    arc_length = abs(angle_diff) * turn_radius
    steps = max(1, int(arc_length / step_size))

    x_list, y_list, yaw_list = [sx], [sy], [syaw]
    direction_list = [1]

    for i in range(1, steps + 1):
        progress = i / steps
        current_yaw = syaw + angle_diff * progress

        # 简化的圆弧计算
        dx = step_size * math.cos(current_yaw)
        dy = step_size * math.sin(current_yaw)

        x_list.append(x_list[-1] + dx)
        y_list.append(y_list[-1] + dy)
        yaw_list.append(current_yaw)
        direction_list.append(1)

    class SimplePath:
        def __init__(self, x, y, yaw, directions):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.directions = directions

    return SimplePath(x_list, y_list, yaw_list, direction_list)


def analystic_expantion(node, ngoal, P):
    """改进的分析性扩展 - 分阶段使用不同策略"""
    # 只有在接近目标时才使用Reed-Shepp曲线
    if is_near_goal(node, ngoal):
        return analystic_expantion_rs(node, ngoal, P)
    else:
        # 远离目标时使用简单路径
        return analystic_expantion_simple(node, ngoal, P)


def analystic_expantion_simple(node, ngoal, P):
    """简单路径扩展 - 用于远距离接近"""
    try:
        simple_path = generate_simple_path(node, ngoal)

        # 检查路径可行性
        if len(simple_path.x) < 2:
            return None

        # 采样检查碰撞
        check_indices = range(0, len(simple_path.x), max(1, len(simple_path.x) // 10))
        pathx = [simple_path.x[i] for i in check_indices]
        pathy = [simple_path.y[i] for i in check_indices]
        pathyaw = [simple_path.yaw[i] for i in check_indices]

        if not is_collision(pathx, pathy, pathyaw, P):
            return simple_path

    except Exception as e:
        pass

    return None


def analystic_expantion_rs(node, ngoal, P):
    """Reed-Shepp扩展 - 仅用于精确泊车"""
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    # 优先选择方向变化少的路径
    sorted_paths = sorted(paths, key=lambda p: calc_rs_path_cost_simple(p))

    for path in sorted_paths[:3]:  # 只检查前3个最优路径，提高效率
        ind = range(0, len(path.x), max(1, C.COLLISION_CHECK_STEP))
        pathx = [path.x[k] for k in ind if k < len(path.x)]
        pathy = [path.y[k] for k in ind if k < len(path.y)]
        pathyaw = [path.yaw[k] for k in ind if k < len(path.yaw)]

        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None


def calc_rs_path_cost_simple(rspath):
    """简化的RS路径代价计算"""
    cost = 0.0
    direction_changes = 0

    # 基础长度代价
    for lr in rspath.lengths:
        if lr >= 0:
            cost += abs(lr)
        else:
            cost += abs(lr) * C.BACKWARD_COST

    # 方向变化统计
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            direction_changes += 1

    # 严厉惩罚方向变化
    cost += direction_changes * C.GEAR_COST * 5

    return cost


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    """更新节点的分析性扩展"""
    path = analystic_expantion(n_curr, ngoal, P)

    if not path:
        return False, None

    # 处理路径数据
    if hasattr(path, 'x') and len(path.x) > 2:
        fx = path.x[1:-1]
        fy = path.y[1:-1]
        fyaw = path.yaw[1:-1]
        fd = path.directions[1:-1] if hasattr(path, 'directions') else [1] * len(fx)
    else:
        return False, None

    # 计算路径代价
    if hasattr(path, 'lengths'):  # RS路径
        fcost = n_curr.cost + calc_rs_path_cost_simple(path)
    else:  # 简单路径
        fcost = n_curr.cost + len(fx) * C.MOVE_STEP

    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


def is_same_grid(node1, node2):
    return (node1.xind == node2.xind and
            node1.yind == node2.yind and
            node1.yawind == node2.yawind)


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    """改进的有效性检查"""
    # 边界检查
    if xind <= P.minx or xind >= P.maxx or yind <= P.miny or yind >= P.maxy:
        return False

    # 碰撞检查 - 适度采样
    if not xlist or not ylist or not yawlist:
        return False

    check_step = max(1, min(C.COLLISION_CHECK_STEP, len(xlist) // 3))
    ind = range(0, len(xlist), check_step)

    nodex = [xlist[k] for k in ind if k < len(xlist)]
    nodey = [ylist[k] for k in ind if k < len(ylist)]
    nodeyaw = [yawlist[k] for k in ind if k < len(yawlist)]

    return not is_collision(nodex, nodey, nodeyaw, P)


def extract_path(closed, ngoal, nstart):
    """提取路径"""
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        if hasattr(node, 'x') and node.x:
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        if node.pind not in closed:
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    if direc:
        direc[0] = direc[1] if len(direc) > 1 else 1
    else:
        direc = [1]

    return Path(rx, ry, ryaw, direc, cost)


def calc_next_node(n_curr, c_id, u, d, P):
    """改进的节点扩展"""
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

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    # 改进的代价计算
    cost = 0.0
    direction = 1 if d > 0 else -1

    if direction > 0:
        cost += abs(step)
    else:
        cost += abs(step) * C.BACKWARD_COST

    # 方向变化惩罚
    if direction != n_curr.direction:
        cost += C.GEAR_COST * 1.5  # 适中的惩罚

    # 转向惩罚
    cost += C.STEER_ANGLE_COST * abs(u)
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)

    total_cost = n_curr.cost + cost
    directions = [direction for _ in range(len(xlist))]

    return Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, total_cost, c_id)


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso, dy_x=None, dy_y=None, dy_yaw=None):
    """
    改进的混合A*规划 - 分阶段策略
    """
    print(f"开始规划: ({sx:.0f},{sy:.0f},{math.degrees(syaw):.0f}°) -> ({gx:.0f},{gy:.0f},{math.degrees(gyaw):.0f}°)")

    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    # 构建环境
    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    # 安全的启发式地图
    hmap = None
    try:
        hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    except Exception as e:
        print(f"使用简化启发式: {e}")
        hmap = [[math.hypot((i + P.minx) * P.xyreso - gx, (j + P.miny) * P.xyreso - gy) / 100
                 for j in range(P.yw)] for i in range(P.xw)]

    # 改进的运动集合
    steer_set, direc_set = calc_motion_set()
    open_set = {calc_index(nstart, P): nstart}
    closed_set = {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P, ngoal))

    iteration_count = 0
    max_iterations = 6000

    while True:
        if not open_set:
            print(f"搜索失败，迭代: {iteration_count}")
            return None

        if iteration_count >= max_iterations:
            print(f"超时，迭代: {max_iterations}")
            return None

        ind = qp.get()
        if ind not in open_set:
            continue

        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        del open_set[ind]

        iteration_count += 1

        if iteration_count % 300 == 0:
            print(f"迭代 {iteration_count}, 开放集: {len(open_set)}")

        # 尝试到达目标
        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)
        if update:
            print(f"成功! 迭代: {iteration_count}")
            return extract_path(closed_set, fpath, nstart)

        # 节点扩展
        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P, ngoal))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P, ngoal))

    return None