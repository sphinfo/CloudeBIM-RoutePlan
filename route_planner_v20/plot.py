import matplotlib # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from shapely import LineString # type: ignore
from route_planner_v20.block import Block


def plot_polygon(coords, ax, blname, yn_value, face_color: str = None):
    centroid = np.mean(coords, axis=0)
    color = face_color if face_color is not None else 'cyan' if yn_value == 'Y' else 'white'
    polygon = plt.Polygon(coords, edgecolor='black', facecolor=color, alpha=0.5)
    ax.add_patch(polygon)
    ax.text(centroid[0], centroid[1], blname, ha='center', va='center', fontsize=9, color='black')


def polygon(block_items: dict):
    # 플롯 초기화
    fig, ax = plt.subplots(figsize=(20, 20))

    # 폴리곤 그리기
    x_min, x_max, y_min, y_max = None, None, None, None

    for blocks in block_items.values():
        for block in blocks.values():
            coords = []
            for i in range(1, 6): 
                x, y = map(lambda x: block.get(f'{x}{i}'), ['x', 'y']) 
                if x is not None and y is not None and x > 1 and y > 1:
                    coords.append([x, y])

                    if x_min is not None:
                        x_min = x_min if x_min < x else x
                        x_max = x_max if x_max > x else x
                    else:
                        x_min, x_max = x, x
                    
                    if y_min is not None:
                        y_min = y_min if y_min < y else y
                        y_max = y_max if y_max > y else y
                    else:
                        y_min, y_max = y, y

            if len(coords) >= 3:
                coords = np.array(coords)
                plot_polygon(coords, ax, block['block_name'], block['yn'])
    
    # print(f'x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max},')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Full Path Visualization')

    plt.draw()
    plt.pause(0.1)
    plt.ion()
    plt.show()
    return ax



def __block(ax, block: dict, face_color = None):
    coords = []
    for i in range(1, 6): 
        x, y = map(lambda x: block.get(f'{x}{i}'), ['x', 'y']) 
        if x is not None and y is not None:
            coords.append([x, y])

    if len(coords) >= 3:
        coords = np.array(coords)
        plot_polygon(coords, ax, block['block_name'], block['yn'], face_color)


def __polygon(ax, block_items: dict):
    # 이전 경로 및 기타 내용 초기화
    ax.cla()

    # 폴리곤 그리기
    x_min, x_max, y_min, y_max = None, None, None, None

    for blocks in block_items.values():
        for block in blocks.values():
            coords = []
            for i in range(1, 6): 
                x, y = map(lambda x: block.get(f'{x}{i}'), ['x', 'y']) 
                if x is not None and y is not None and x > 1 and y > 1:
                    coords.append([x, y])

                    if x_min is not None:
                        x_min = x_min if x_min < x else x
                        x_max = x_max if x_max > x else x
                    else:
                        x_min, x_max = x, x
                    
                    if y_min is not None:
                        y_min = y_min if y_min < y else y
                        y_max = y_max if y_max > y else y
                    else:
                        y_min, y_max = y, y

            if len(coords) >= 3:
                coords = np.array(coords)
                plot_polygon(coords, ax, block['block_name'], block['yn'])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Full Path Visualization')


def __get_color(current_direction: int, allocate_cell_name: str):
    # # 할당셀 별로 색 다르게
    # if allocate_cell_name.startswith('OL'):
    #     color = 'purple' if current_direction == 1 else 'chocolate'
    # else:
    #     if int(allocate_cell_name.split('_')[-1]) % 2 == 1:
    #         color = 'lightgreen' if current_direction == 1 else 'red'
    #     else:
    #         color = 'blue' if current_direction == 1 else 'gold'
    # return color
    return 'lightgreen' if current_direction == 1 else 'red'

def full(ax, block_items: dict, route_plan: list, index: int = 1):
    __polygon(ax, block_items)
    # plt.pause(10)
    # previous_direction = None
    for i in range(1, len(route_plan)):
        if i < index:
            continue
        x_prev, y_prev = float(route_plan[i - 1]['x']), float(route_plan[i - 1]['y'])
        x_curr, y_curr = float(route_plan[i]['x']), float(route_plan[i]['y'])
        current_direction = route_plan[i]['direction']

        print(f'{i} x_curr: {x_curr}, y_curr: {y_curr}, current_direction: {current_direction}, cell: {route_plan[i]["cell_name"]}')
        # 전진 light green, 후진 red

        if any([coord is None for coord in [x_prev, y_prev, x_curr, y_curr]]):
            print(f'경로 값을 확인해주세요. Timeline: {i}')
            break
    
        color = __get_color(current_direction, route_plan[i]['allocate_cell_name'])
        ax.plot([x_prev, x_curr], [y_prev, y_curr], marker='o', linestyle='-', color=color)

        plt.pause(0.01)

def timeline_next(ax, block_items: dict, route_plan: list, index: int):
    __polygon(ax, block_items)
    return next(ax, route_plan, index)

def draw_coord(ax, block_items: dict, block_name: str):
    __polygon(ax, block_items)
    block = Block.get_block_by_name(block_items, block_name)
    __block(ax, block, 'red')

    # 전방 lightgreen
    ax.plot(float(block['x_t']), float(block['y_t']), marker='o', linestyle='-', color='lightgreen')

    # 후방 red
    ax.plot(float(block['x_b']), float(block['y_b']), marker='o', linestyle='-', color='red')

    for i in range(1, 6): 
        x, y = map(lambda x: block.get(f'{x}{i}'), ['x', 'y']) 
        if x is not None and y is not None:
            ax.plot(float(x), float(y), marker='o', color='black')
    plt.pause(0.3)



def next(ax, route_plan: list, index: int):
    if index + 1 >= len(route_plan):
        print('값을 초과 하였습니다. 올바른 값을 입력해주세요.')
        return index

    x_prev, y_prev = float(route_plan[index]['x']), float(route_plan[index]['y'])
    x_curr, y_curr = float(route_plan[index + 1]['x']), float(route_plan[index + 1]['y'])
    current_direction = route_plan[index + 1]['direction']

    print(f'{index} x_curr: {x_curr}, y_curr: {y_curr}, current_direction: {current_direction}, cell: {route_plan[index]["cell_name"]}->{route_plan[index+1]["cell_name"]}')

    if any([int(coord) < 1 for coord in [x_prev, y_prev, x_curr, y_curr]]):
        print(f'경로 값을 확인해주세요. x_prev: {x_prev}, y_prev: {y_prev}, x_curr: {x_curr}, y_curr: {y_curr}')
        return index

    color = __get_color(current_direction, route_plan[index + 1]['allocate_cell_name'])
    ax.plot([x_prev, x_curr], [y_prev, y_curr], marker='o', linestyle='-', color=color)

    plt.pause(0.1)
    return index + 1

def prev(ax, route_plan: list, index: int):
    if index - 1 < 0:
        print('이전 경로가 없습니다. 올바른 값을 입력해주세요.')
        return index

    x_prev, y_prev = float(route_plan[index - 1]['x']), float(route_plan[index - 1]['y'])
    x_curr, y_curr = float(route_plan[index]['x']), float(route_plan[index]['y'])
    current_direction = route_plan[index]['direction']

    if any([int(coord) < 1 for coord in [x_prev, y_prev, x_curr, y_curr]]):
        print(f'경로 값을 확인해주세요. x_prev: {x_prev}, y_prev: {y_prev}, x_curr: {x_curr}, y_curr: {y_curr}')
        return index

    color = __get_color(current_direction, route_plan[index + 1]['allocate_cell_name'])
    ax.plot([x_prev, x_curr], [y_prev, y_curr], marker='o', linestyle='-', color=color)

    plt.pause(0.1)
    return index - 1

def alloc(ax, block_items: dict, route_plan: list, allocate_cell_name: str):
    __polygon(ax, block_items)

    target = [v for v in route_plan if v['allocate_cell_name'] == allocate_cell_name]

    if not target:
        print(f'올바른 할당셀을 입력하세요. 할당셀: {", ".join(list(set([v["allocate_cell_name"] for v in route_plan])))}')
        plt.pause(0.1)
        return

    for i in range(1, len(target)):
        x_prev, y_prev = float(target[i - 1]['x']), float(target[i - 1]['y'])
        x_curr, y_curr = float(target[i]['x']), float(target[i]['y'])
        current_direction = target[i]['direction']

        if any([int(coord) < 1 for coord in [x_prev, y_prev, x_curr, y_curr]]):
            print(f'경로 값을 확인해주세요. x_prev: {x_prev}, y_prev: {y_prev}, x_curr: {x_curr}, y_curr: {y_curr}')
            break
    
        color = __get_color(current_direction, route_plan[i + 1]['allocate_cell_name'])
        ax.plot([x_prev, x_curr], [y_prev, y_curr], marker='o', linestyle='-', color=color)
    plt.pause(0.1)


def draw_obstacle(block_items: dict, route_plan: list, intersected_blocks: list, param: dict):
    ax = polygon(block_items)
    already_blocks = []
    for block_name, idx in intersected_blocks:
        block = Block.get_block_by_name(block_items, block_name)
        if block not in already_blocks:
            # print(f'idx: {idx}, {route_plan[idx]}')
            # print(f'idx: {idx + 1}, {route_plan[idx]}')
            # print(f'block: {block_name}')
            if block_name not in already_blocks:
                __block(ax, block, 'red')
            next(ax, route_plan, idx)
            line = LineString([(float(route_plan[idx]['x']), float(route_plan[idx]['y'])), (float(route_plan[idx + 1]['x']), float(route_plan[idx + 1]['y']))])
            poly = plt.Polygon(list(line.buffer(param['gap'], cap_style='flat').exterior.coords), edgecolor='gold', facecolor='gold', alpha=0.2)
            ax.add_patch(poly)
        already_blocks.append(block_name)
        plt.pause(0.2)
    while True:
        match input("경로 생성 불가, 종료: '0': "):
            case '0':
                print("종료")
                plt.close()
                break
        

def draw(block_items: dict, route_plan: list):
    ax = polygon(block_items)
    index = 0
        
    while True:
        match input(f"n: 다음, p: 이전, 타임라인 이동: t, 할당셀 그리기: a, 경로 전체 보기: 'f'  현재index({index})부터 그리기: 'i', 블럭 좌표 확인: b, 종료: '0': "):
            case 'p':
                print("이전")
                index = prev(ax, route_plan, index)
            case 'n':
                print("다음")
                index = next(ax, route_plan, index)
            case 'f':
                print("경로 전체 보기")
                full(ax, block_items, route_plan)
            case 'i':
                print("현재 index 부터 그리기")
                full(ax, block_items, route_plan, index)
            case 't':
                timeline = input("타임라인을 입력하세요: ")
                index = timeline_next(ax, block_items, route_plan, int(timeline))
            case 'a':
                allocate_cell_name = input("할당셀을 입력하세요(ex: AL_2_1, OL_1_1): ")
                alloc(ax, block_items, route_plan, allocate_cell_name)
            case 'b':
                block_name = input("블럭명을 입력하세요: ")
                draw_coord(ax, block_items, block_name)
            case '0':
                print("종료")
                plt.close()
                break
            case _:
                print("유효한 값을 입력하세요.")
        
        plt.pause(0.1)