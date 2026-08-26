"""Convert the SolidWorks URDF to MJCF while keeping the proven leg conventions."""

from __future__ import annotations

import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE.parent
URDF = ASSET_DIR / "urdf" / "00-a1-all_0819_asm.urdf"
MESH_DIR = ASSET_DIR / "meshes"
REFERENCE = ASSET_DIR.parent / "legs_narrow" / "mjcf" / "legs_narrow.xml"
OUTPUT = HERE / "nlegs_body.xml"


def _compile_urdf() -> ET.ElementTree:
    text = URDF.read_text()
    compiler = (
        f'<mujoco><compiler meshdir="{MESH_DIR}" strippath="true" '
        'discardvisual="false" balanceinertia="true" fusestatic="false"/></mujoco>'
    )
    index = text.index(">", text.index("<robot")) + 1
    text = text[:index] + compiler + text[index:]
    with tempfile.TemporaryDirectory() as tmp:
        urdf = Path(tmp) / "nlegs_body.urdf"
        xml = Path(tmp) / "nlegs_body.xml"
        urdf.write_text(text)
        model = mujoco.MjModel.from_xml_path(str(urdf))
        mujoco.mj_saveLastXML(str(xml), model)
        return ET.parse(xml)


def _canonicalize(tree: ET.ElementTree) -> None:
    root = tree.getroot()
    reference = ET.parse(REFERENCE).getroot()
    canonical_joints = {joint.get("name"): joint for joint in reference.findall(".//joint")}

    root.set("model", "nlegs_body")
    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "../meshes")

    worldbody = root.find("worldbody")
    base = worldbody.find("body")
    base.set("name", "base")
    base.insert(0, ET.Element("freejoint", {"name": "base_freejoint"}))
    base.insert(1, ET.Element("site", {"name": "imu", "pos": "0 0 0", "quat": "1 0 0 0", "group": "3"}))

    for joint in root.findall(".//joint"):
        side, number = joint.get("name").split("-joint0")
        name = f"joint_{side}{number}"
        source = canonical_joints[name]
        joint.set("name", name)
        for key in ("axis", "range"):
            joint.set(key, source.get(key))
        joint.set("limited", "true")
        joint.set("damping", "0")
        joint.set("frictionloss", "0")

    # The CAD URDF duplicates each triangle mesh as visual and collision geometry.
    # Keep one non-colliding visual mesh per link and use the proven foot capsules.
    for body in root.findall(".//body"):
        seen = set()
        for geom in list(body.findall("geom")):
            key = (geom.get("type", "sphere"), geom.get("mesh"), geom.get("pos"), geom.get("quat"))
            if key in seen:
                body.remove(geom)
                continue
            seen.add(key)
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "1")

    for foot_name in ("Link_R6", "Link_L6"):
        foot = root.find(f'.//body[@name="{foot_name}"]')
        for y in (0.03, 0.0, -0.03):
            ET.SubElement(
                foot,
                "geom",
                {
                    "type": "capsule",
                    "fromto": f"-0.04 {y:g} -0.0015 0.14 {y:g} -0.0015",
                    "size": "0.012",
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "3",
                    "rgba": "0.9373 0.2667 0.2667 1",
                },
            )

    for tag in ("actuator", "sensor"):
        old = root.find(tag)
        if old is not None:
            root.remove(old)
        root.append(copy.deepcopy(reference.find(tag)))


def _validate() -> None:
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    expected = ["base_freejoint"] + [f"joint_R{i}" for i in range(1, 7)] + [f"joint_L{i}" for i in range(1, 7)]
    assert names == expected, (names, expected)
    assert model.nq == 19 and model.nv == 18 and model.nu == 12
    print(f"OK: {OUTPUT}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print("joints:", names)


def main() -> None:
    tree = _compile_urdf()
    _canonicalize(tree)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    _validate()


if __name__ == "__main__":
    main()
