import json


def make_map(map_path):

    with open(map_path,'r',encoding='UTF-8') as f:
        map=json.load(f)
    ox = map['obstacles']['ox']
    oy = map['obstacles']['oy']
    sp = map['start_position']
    gp = map['parking_sport']
    return ox, oy, sp, gp

def get_obs(map_path_dyobs):
    with open(map_path_dyobs, 'r', encoding='UTF-8') as f:
        map_dyobs = json.load(f)
    dyobs_x = map_dyobs['X']
    dyobs_y = map_dyobs['Y']
    dyobs_yaw = map_dyobs['Yaw']
    return dyobs_x, dyobs_y, dyobs_yaw
