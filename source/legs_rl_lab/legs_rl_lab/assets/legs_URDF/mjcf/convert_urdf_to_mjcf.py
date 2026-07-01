"""Convert legs_URDF.urdf -> MuJoCo MJCF for sim2sim, mirroring A1_legs_V1.xml.

Strategy: let MuJoCo's URDF compiler parse the kinematics (link transforms,
joint axes/signs, inertials) from the URDF, then inject the pieces sim2sim.py
needs (floating base, IMU site + sensors, torque motors, ground plane).
"""
import os
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_URDF = "/home/woan/workspace/robot_assets/legs_URDF/urdf/legs_URDF.urdf"
MESH_DIR = os.path.join(HERE, "..", "meshes")
TMP_URDF = os.path.join(HERE, "_legs_URDF_mjc.urdf")
OUT_XML = os.path.join(HERE, "legs_URDF.xml")

# joints in MuJoCo dof order (right leg first, then left), as the URDF tree dictates
R_JOINTS = [f"joint_R{i}" for i in range(1, 7)]
L_JOINTS = [f"joint_L{i}" for i in range(1, 7)]
JOINTS = R_JOINTS + L_JOINTS
# effort limits mirror legs.py / A1: hip..knee (1-4)=27, ankle (5-6)=7
FORCE = {**{f"joint_R{i}": 27 for i in range(1, 5)}, **{f"joint_R{i}": 7 for i in range(5, 7)},
         **{f"joint_L{i}": 27 for i in range(1, 5)}, **{f"joint_L{i}": 7 for i in range(5, 7)}}

# --- 1. inject a <mujoco><compiler> block so MuJoCo resolves meshes correctly ---
with open(SRC_URDF, "r") as f:
    urdf = f.read()
inject = (
    f'<mujoco>\n'
    f'  <compiler meshdir="{MESH_DIR}" strippath="true" discardvisual="false" '
    f'balanceinertia="true" fusestatic="false"/>\n'
    f'</mujoco>\n'
)
# place right after the opening <robot ...> tag
idx = urdf.index(">", urdf.index("<robot")) + 1
urdf = urdf[:idx] + "\n" + inject + urdf[idx:]
with open(TMP_URDF, "w") as f:
    f.write(urdf)

# --- 2. compile via MuJoCo and dump the auto-converted MJCF ---
model = mujoco.MjModel.from_xml_path(TMP_URDF)
mujoco.mj_saveLastXML(OUT_XML, model)

# --- 3. post-process the saved MJCF to add sim2sim-required elements ---
tree = ET.parse(OUT_XML)
root = tree.getroot()
root.set("model", "legs_URDF")

compiler = root.find("compiler")
compiler.set("meshdir", "../meshes")
compiler.set("angle", "radian")

worldbody = root.find("worldbody")
base = worldbody.find("body")  # first body == root link (base_link)

# floating base + IMU site
fj = ET.Element("freejoint", {"name": "base_freejoint"})
base.insert(0, fj)
site = ET.Element("site", {"name": "imu", "pos": "0 0 0", "quat": "1 0 0 0", "group": "3"})
base.insert(1, site)

# torque motors (control_mode == "motor" in sim2sim) in R..L order
act = ET.SubElement(root, "actuator")
for j in JOINTS:
    ET.SubElement(act, "motor", {"name": j, "joint": j, "forcerange": f"-{FORCE[j]} {FORCE[j]}"})

# sensors mirroring A1: jointpos/vel/torque + IMU
sen = ET.SubElement(root, "sensor")
for j in L_JOINTS + R_JOINTS:
    ET.SubElement(sen, "jointpos", {"name": f"{j}_pos", "joint": j})
for j in L_JOINTS + R_JOINTS:
    ET.SubElement(sen, "jointvel", {"name": f"{j}_vel", "joint": j})
for j in L_JOINTS + R_JOINTS:
    ET.SubElement(sen, "jointactuatorfrc", {"name": f"{j}_torque", "joint": j})
ET.SubElement(sen, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"})
ET.SubElement(sen, "gyro", {"name": "imu_gyro", "site": "imu"})
ET.SubElement(sen, "accelerometer", {"name": "imu_acc", "site": "imu"})
ET.SubElement(sen, "framepos", {"name": "frame_pos", "objtype": "site", "objname": "imu"})
ET.SubElement(sen, "framelinvel", {"name": "frame_vel", "objtype": "site", "objname": "imu"})

# ground plane + light + skybox (scene dressing, like A1)
asset = root.find("asset")
ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "flat", "rgb1": "0 0 0",
                                 "rgb2": "0 0 0", "width": "512", "height": "3072"})
ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker",
                                 "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
                                 "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"})
ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane",
                                  "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
ET.SubElement(worldbody, "light", {"pos": "1 0 3.5", "dir": "0 0 -1", "directional": "true"})
ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane",
                                  "material": "groundplane"})

ET.indent(tree, space="  ")
tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)
os.remove(TMP_URDF)

# --- 4. validate: reload the final XML and report structure ---
m = mujoco.MjModel.from_xml_path(OUT_XML)
jnt_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("OK ->", OUT_XML)
print("nq=%d nv=%d nu=%d" % (m.nq, m.nv, m.nu))
print("joints:", jnt_names)
print("actuators:", act_names)
print("dof(qpos[7:]) joint order:", jnt_names[1:])  # after freejoint
