import genesis as gs

# Genesis 초기화
gs.init()

# Scene 생성
scene = gs.Scene(show_viewer=True)

scene.add_entity(morph=gs.morphs.Plane())
# 색상 리스트와 위치 지정
cube_specs = [
    {"pos": (-0.0, 0.0, 1.0), "color": (1.0, 0.0, 0.0), "name": "cube_red"},  # 빨강
    {"pos": ( 0.0, 0.0, 0.5), "color": (0.0, 1.0, 0.0), "name": "cube_green"},  # 초록
    {"pos": ( 0.0, 0.0, 1.5), "color": (0.0, 0.0, 1.0), "name": "cube_blue"},  # 파랑
]

# 각 색상 큐브 생성
cube_red = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(-0.0, 0.0, 1.0),
        size=(0.1, 0.1, 0.1),
        collision=True
    ),
    material=gs.materials.Rigid(rho=1000.0),
    surface=gs.surfaces.Default(
        color=(1.0, 0.0, 0.0),
        vis_mode='visual'
    )
)
cube_green = scene.add_entity(
    morph=gs.morphs.Box(
        pos=( 0.0, 0.0, 0.5),
        size=(0.1, 0.1, 0.1),
        collision=True
    ),
    material=gs.materials.Rigid(rho=1000.0),
    surface=gs.surfaces.Default(
        color=(0.0, 1.0, 0.0),
        vis_mode='visual'
    )
)
cube_blue = scene.add_entity(
    morph=gs.morphs.Box(
        pos=( 0.0, 0.0, 1.5),
        size=(0.1, 0.1, 0.1),
        collision=True
    ),
    material=gs.materials.Rigid(rho=1000.0),
    surface=gs.surfaces.Default(
        color=(0.0, 0.0, 1.0),
        vis_mode='visual'
    )
)

# Scene 빌드 및 실행
scene.build()

cube_red.get_pos()
cube_green.get_pos()
cube_blue.get_pos()

pos_1 = list()
pos_2 = list()
pos_3 = list()


for _ in range(100):

    pos1 = cube_red.get_pos()
    pos2 = cube_green.get_pos()
    pos3 = cube_blue.get_pos()
    pos_1.append(pos1)
    pos_2.append(pos2)
    pos_3.append(pos3)
    scene.step()

print("pos_1", pos_1)
print("pos_2", pos_2)
print("pos_3", pos_3)