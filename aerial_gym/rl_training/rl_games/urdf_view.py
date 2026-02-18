from urdfpy import URDF

robot = URDF.load("/home/control-lab/aerial-gym-docker/aerialgym_ws_v2/src/aerial_gym_simulator/resources/robots/x500arm/model.urdf")
print("Links:", len(robot.links))
print("Joints:", len(robot.joints))

robot.show()