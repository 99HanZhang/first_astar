"""

 仅绘制地图
"""
import matplotlib.pyplot as plt
import input.make_map as mp

def map_display(map_scene):
    picture_scene =  map_scene.replace('/', '/').replace('.json', '.jpg')
    picture = plt.imread(picture_scene)
    ox, oy,sp,gp = mp.make_map(map_scene)
    plt.rcParams['xtick.direction'] = 'in'
    plt.cla()
    plt.plot(ox, oy, ",k")
    plt.tick_params(axis='x', direction='in', top=True, bottom=False, labelbottom=False, labeltop=True)
    plt.axis("equal")
    plt.imshow(picture, extent=[min(ox), max(ox), min(oy), max(oy)])
    print("绘制地图结束!")
    plt.show()

