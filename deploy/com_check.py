#!/usr/bin/env python3
"""nlegs_body 质心 / 支撑多边形静态稳定性分析(纯离线, 不碰硬件).

以 assets/nlegs_body/mjcf/nlegs_body.xml 为唯一真源读取质量与几何, 对给定的 12 关节角
(实机序 L1..L6,R1..R6, sim 符号约定 —— 与 pose_move.py 下发/读回的数组同一约定)计算:
  - 双脚贴平地面所需的 base 俯仰/横滚(这就是 IMU 的参考轨迹)
  - 整机质心水平投影
  - 支撑多边形(双脚 capsule 触地点凸包)与稳定边距
用途: 判断"走路准备姿态 <-> 下蹲平衡姿态"轨迹上质心何时越界、往哪个方向越界。

约定说明: 越限判据以**实机能到的角度**为准, 取 rl_real_py/configs/common.yaml 的
发布安全限位(它处处 ≤ XML range: 髋pitch/yaw 紧得多, 膝/踝 两边相同)。
超出只告警不夹断 —— 分析要看真实姿态。
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.normpath(os.path.join(
    HERE, "..", "source", "legs_rl_lab", "legs_rl_lab",
    "assets", "nlegs_body", "mjcf", "nlegs_body.xml"))
BALANCE_JSON = os.path.join(HERE, "pose_logs", "balance_pose.json")
COMMON_YAML = os.path.join(HERE, "rl_real_py", "configs", "common.yaml")

REAL = ["L1", "L2", "L3", "L4", "L5", "L6", "R1", "R2", "R3", "R4", "R5", "R6"]
DESC = ["髋pitch", "髋roll", "髋yaw", "膝", "踝pitch", "踝roll"] * 2
STAND = np.array([-0.1, 0., 0., 0.2, -0.1, 0., -0.1, 0., 0., 0.2, -0.1, 0.])
ANKLE_PITCH_LIM = 0.4                       # 踝pitch 行程(与 common.yaml 的 L5/R5 一致)
SEG_SHAPE = [1, 2, 5, 7, 8, 11]             # 髋roll/髋yaw/踝roll
SEG_LIFT = [0, 3, 4, 6, 9, 10]              # 髋pitch/膝/踝pitch

# 蹭脚序列的分组: 矢状链(脚不需要挪位) / 左右各自的额面链(蹭脚时只动一侧)
SPLAY_L = [1, 2, 5]                         # L2 髋roll, L3 髋yaw, L6 踝roll
SPLAY_R = [7, 8, 11]
LEAN_HIP = [1, 7]                           # 两腿髋roll 同向 -> 骨盆纯侧移
LEAN_ANK = [5, 11]                          # 踝roll 反向抵消 -> 脚底仍贴平
G = 9.81


def real_limits(path=COMMON_YAML):
    """实机可达行程(实机序, rad): rl_real_py 下发前真正会 clip 的那一层。

    比 XML 的 range 可信: 髋pitch ±1.05(XML 写 -3.14)、髋yaw ±1.0(XML 写 ±2.75)。
    髋roll 左右不对称但物理对称: 内收侧都只有 0.26 rad(15°, 腿撞机身),
    外展侧到 1.05 —— 这个 0.26 就是侧倾幅度的硬天花板。
    """
    with open(path) as f:
        c = yaml.safe_load(f)
    order = c["joint_index_in_real"]
    lo = dict(zip(order, c["joint_lower_limits"]))
    hi = dict(zip(order, c["joint_upper_limits"]))
    return (np.array([lo[n] for n in REAL]), np.array([hi[n] for n in REAL]))


# ---------------------------------------------------------------- 数学工具
def quat2mat(q):
    """MuJoCo 四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_rot(axis, ang):
    """绕单位轴 axis 转 ang 的旋转矩阵(Rodrigues)。"""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def align_rot(v_from, v_to):
    """把 v_from 转到 v_to 的最小旋转(用于求贴平地面所需的 base 姿态)。"""
    a = np.asarray(v_from, float) / np.linalg.norm(v_from)
    b = np.asarray(v_to, float) / np.linalg.norm(v_to)
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    if s < 1e-12:
        return np.eye(3) if a @ b > 0 else -np.eye(3)
    return axis_rot(c / s, np.arctan2(s, a @ b))


def convex_hull(pts):
    """2D 凸包(Andrew monotone chain), 返回逆时针顶点。"""
    p = sorted(map(tuple, np.asarray(pts)))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def margin_to_hull(pt, hull):
    """点到凸包边界的距离; 内部为正, 外部为负。同时返回最近边的外法向。"""
    best, bn = 1e9, None
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-12:
            continue
        nor = np.array([e[1], -e[0]]) / L       # 逆时针 -> 该法向朝外
        d = nor @ (pt - a)                      # >0 表示在这条边外侧
        if -d < best:
            best, bn = -d, nor
    return best, bn


# ---------------------------------------------------------------- 模型解析
class Model:
    """从 MJCF 读出的刚体树: 每个 body 的 父级/偏移/关节轴/质量/质心。"""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        wb = root.find("worldbody")
        self.bodies = {}          # name -> dict(parent,pos,quat,joint,axis,range,mass,com)
        self.order = []           # 保证父在子前
        self.feet = {}            # name -> [(p_end_a, p_end_b, radius), ...]
        self._walk(wb.find("body"), None)
        self.mass_total = sum(b["mass"] for b in self.bodies.values())

    def _walk(self, elem, parent):
        name = elem.get("name")
        pos = np.array([float(v) for v in elem.get("pos", "0 0 0").split()])
        quat = [float(v) for v in elem.get("quat", "1 0 0 0").split()]
        jn = elem.find("joint")
        ine = elem.find("inertial")
        self.bodies[name] = {
            "parent": parent,
            "pos": pos,
            "R0": quat2mat(quat),
            "joint": jn.get("name") if jn is not None else None,
            "axis": (np.array([float(v) for v in jn.get("axis").split()])
                     if jn is not None else None),
            "range": ([float(v) for v in jn.get("range").split()]
                      if jn is not None and jn.get("range") else None),
            "mass": float(ine.get("mass")) if ine is not None else 0.0,
            "com": (np.array([float(v) for v in ine.get("pos").split()])
                    if ine is not None else np.zeros(3)),
        }
        self.order.append(name)
        caps = [g for g in elem.findall("geom") if g.get("type") == "capsule"]
        if caps:
            self.feet[name] = [
                (np.array([float(v) for v in g.get("fromto").split()[:3]]),
                 np.array([float(v) for v in g.get("fromto").split()[3:]]),
                 float(g.get("size")))
                for g in caps]
        for ch in elem.findall("body"):
            self._walk(ch, name)

    def fk(self, qmap):
        """在 base 系(base 位于原点、姿态为单位)做正运动学。返回 name -> (R, p)。"""
        T = {}
        for name in self.order:
            b = self.bodies[name]
            Rl = b["R0"]
            if b["joint"] is not None:
                Rl = Rl @ axis_rot(b["axis"], qmap[b["joint"]])
            if b["parent"] is None:
                T[name] = (Rl, b["pos"].copy())
            else:
                Rp, pp = T[b["parent"]]
                T[name] = (Rp @ Rl, pp + Rp @ b["pos"])
        return T


# ---------------------------------------------------------------- 姿态求解
def _pose_at(m, T, pitch, roll):
    """给定 base 俯仰/横滚, 返回 (Rb, 质心, 触地候选点, 点所属脚, 落地高度 z0)。

    base 高度由"最低触地点贴地"唯一确定, 所以无穿透约束自动满足。
    """
    Rb = axis_rot([0, 1, 0], pitch) @ axis_rot([1, 0, 0], roll)
    cand, who = [], []
    for fname, caps in m.feet.items():
        Rf, pf = T[fname]
        for ea, eb, r in caps:
            for e in (ea, eb):
                cand.append(Rb @ (pf + Rf @ e) - np.array([0, 0, r]))
                who.append(fname)
    cand = np.array(cand)
    z0 = cand[:, 2].min()
    cand[:, 2] -= z0

    csum = np.zeros(3)
    for name, b in m.bodies.items():
        if b["mass"] > 0:
            R, p = T[name]
            csum += b["mass"] * (Rb @ (p + R @ b["com"]))
    com = csum / m.mass_total
    com[2] -= z0
    return Rb, com, cand, np.array(who), z0


def rest_orientation(m, T, pitch0):
    """求刚体搁在地面上的静止姿态。

    物理: 高摩擦下准静态搁置的稳定平衡位形 = 质心高度的局部极小。
    必须限定在近直立陆域: 不约束的话"整机倒置"会让质心降到接触平面下方,
    数值上更优但物理上不存在。起点用"脚底贴平"的解析猜测。
    """
    lim_p, lim_r = np.radians(30.0), np.radians(25.0)

    def h(pr):
        if abs(pr[0] - pitch0) > lim_p or abs(pr[1]) > lim_r:
            return np.inf                       # 越出近直立陆域
        z = _pose_at(m, T, pr[0], pr[1])[1][2]
        return np.inf if z <= 0 else z          # 质心必在接触平面上方

    best = min(((h([p, r]), [p, r])
                for p in pitch0 + np.radians(np.arange(-30, 30.1, 2.0))
                for r in np.radians(np.arange(-20, 20.1, 2.0))),
               key=lambda t: t[0])[1]
    step = np.radians(2.0)
    while step > np.radians(0.005):                 # 模式搜索: 命中则前进, 否则减步
        improved = False
        for d in ([step, 0], [-step, 0], [0, step], [0, -step]):
            trial = [best[0] + d[0], best[1] + d[1]]
            if h(trial) < h(best) - 1e-12:
                best, improved = trial, True
                break
        if not improved:
            step *= 0.5
    return best


def analyze(m, q_real, label=""):
    """给定实机序 12 关节角, 求贴平地面时的 base 姿态、质心投影与稳定边距。

    基准假设: 两只脚底都贴在地上 —— 这就是控制器的意图, 踝力矩正是在维持它。
    不用"质心高度极小"去反求落地姿态: 那个判据会跳到远处另一个能量盆(趾脚大
    角度倾斜), 而不是机器人实际所在的那个平衡。

    两脚能不能 同时 贴平是一道独立的合法性检查(incons): 若左右矢状链不一致, 单个
    base 姿态下必然一只压脚尖、一只压脚跟, 此时边距数字本身就不该当真。
    支撑多边形按"每只脚各自贴平"构造(忽略残余倾斜带来的二阶 xy 偏移),
    对左右一致的对称姿态是精确的。
    """
    qmap = {f"joint_{n}": float(v) for n, v in zip(REAL, q_real)}
    T = m.fk(qmap)

    # 左右矢状链之和(髋pitch+膝+踝pitch)就是各自贴平所需的 base 俯仰
    sagL, sagR = -(q_real[0] + q_real[3] + q_real[4]), -(q_real[6] + q_real[9] + q_real[10])
    incons = np.degrees(sagL - sagR)

    zs = [T[f][0][:, 2] for f in ("Link_L6", "Link_R6")]
    Rb = align_rot(np.mean(zs, axis=0), [0, 0, 1])
    tilt = [np.degrees(np.arccos(np.clip((Rb @ z) @ [0, 0, 1], -1, 1))) for z in zs]

    cand, who = [], []
    for fname, caps in m.feet.items():
        Rf, pf = T[fname]
        for ea, eb, r in caps:
            for e in (ea, eb):
                cand.append(Rb @ (pf + Rf @ e) - np.array([0, 0, r]))
                who.append(fname)
    cand = np.array(cand)
    z0 = cand[:, 2].min()
    cand[:, 2] -= z0

    csum = np.zeros(3)
    for name, b in m.bodies.items():
        if b["mass"] > 0:
            R, p = T[name]
            csum += b["mass"] * (Rb @ (p + R @ b["com"]))
    com = csum / m.mass_total
    com[2] -= z0

    # 把水平原点挪到两踝pitch 轴中点: 力矩的力臂、支撑区都必须相对这个支点来读,
    # 而不是相对 base 原点 —— 躯干一倾 base 原点就相对脚平移了。
    anchor = np.mean([(Rb @ T[f][1])[:2] for f in ("Link_L5", "Link_R5")], axis=0)
    com[:2] -= anchor
    cand[:, :2] -= anchor

    hull = convex_hull(cand[:, :2])          # 每只脚各自贴平的意图足印
    mg, nor = margin_to_hull(com[:2], hull)
    return dict(label=label, com=com, hull=hull, margin=mg, normal=nor,
                pitch=np.degrees(np.arctan2(-Rb[2, 0], Rb[2, 2])),
                roll=np.degrees(np.arctan2(Rb[2, 1], Rb[2, 2])),
                tilt=tilt, incons=incons, ncontact=len(cand),
                nLR=(6, 6), hip_h=(Rb @ T["Link_L2"][1])[2] - z0)


KP_ANKLE_PITCH = 40.0                       # armcontrol 生效值(实机序 idx 4/10)
FRICTION_ANKLE = (2.0, 3.0)                 # 实测静摩擦门槛区间 N·m


def torque_view(m, r):
    """准静态双支撑下维持该姿态所需的踝pitch 力矩, 及由此带来的位置误差与回路增益。

    平衡要求压心(CoP)落在质心正下方, 而 τ_踝 = -(CoP_x - x_踝)*Fn, 于是
    两只踝共同承担  τ_total = mg * CoM_x。纯位置 PD 下要出这个力矩, 踝就必须
    偏离指令角  δ = τ/kp。而 δ 直接就是 base 俯仰误差(矢状链之和), 会把质心
    再推出 h*δ —— 构成正反馈。回路增益 = mg*h / kp_total, 达 1 即发散倒地。
    """
    mg = m.mass_total * G
    tau_total = mg * r["com"][0]
    tau_each = tau_total / 2.0
    kp_total = 2.0 * KP_ANKLE_PITCH
    gain = mg * r["com"][2] / kp_total
    delta = tau_each / KP_ANKLE_PITCH
    return dict(tau_each=tau_each, delta=delta, gain=gain,
                amp=(1.0 / (1.0 - gain) if gain < 1 else float("inf")),
                kp_crit=mg * r["com"][2] / 2.0)


def show(r, m=None):
    xs, ys = r["hull"][:, 0], r["hull"][:, 1]
    print(f"\n=== {r['label']} ===")
    print(f"  base 俯仰(脚底贴平所需)  pitch {r['pitch']:+.2f}°  roll {r['roll']:+.2f}°")
    bad = "  !! 两脚无法同时贴平, 下面的边距不作数" if abs(r["incons"]) > 1.0 else ""
    print(f"  左右矢状链不一致 {r['incons']:+.2f}°"
          f"   残余倾斜 L{r['tilt'][0]:.2f}° R{r['tilt'][1]:.2f}°{bad}")
    print(f"  髋高 {r['hip_h']*1000:.0f} mm   质心高 {r['com'][2]*1000:.0f} mm")
    print(f"  质心投影  x {r['com'][0]*1000:+.1f} mm   y {r['com'][1]*1000:+.1f} mm")
    print(f"  支撑多边形 x∈[{xs.min()*1000:+.0f},{xs.max()*1000:+.0f}] "
          f"y∈[{ys.min()*1000:+.0f},{ys.max()*1000:+.0f}] mm")
    tag = "稳定" if r["margin"] > 0 else "!! 已越界(会倒)"
    print(f"  静态稳定边距 {r['margin']*1000:+.1f} mm   {tag}")
    if m is None:
        return
    t = torque_view(m, r)
    f0, f1 = FRICTION_ANKLE
    band = ("淹在静摩擦里(踝位置不可控)" if abs(t["tau_each"]) < f1
            else "高于摩擦门槛")
    print(f"  维持所需踝pitch 力矩 {t['tau_each']:+.2f} N·m/只"
          f"  (静摩擦 {f0:.0f}~{f1:.0f}) -> {band}")
    print(f"  kp={KP_ANKLE_PITCH:.0f} 下踝必然偏离指令 {np.degrees(t['delta']):+.2f}°"
          f"  = base 俯仰误差, 把质心再推 {t['delta']*r['com'][2]*1000:+.1f} mm")
    print(f"  正反馈回路增益 mg*h/kp_total = {t['gain']:.3f}"
          f"  (误差放大 {t['amp']:.2f}x)   临界 kp = {t['kp_crit']:.1f} N·m/rad/只")


def _solve_q1(m, q_of, q4, com_x_target, lo=-1.6, hi=0.5):
    """固定膝角 q4, 二分求 q1 使 CoM_x = com_x_target(CoM_x 对 q1 单调)。"""
    def f(q1):
        return analyze(m, q_of(q1, q4))["com"][0] - com_x_target

    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def com_neutral_family(m, com_x_target=0.020, roll=0.0, yaw=0.0):
    """求"质心中立下蹲族": 躯干垂直 + 双脚贴平 + 质心固定在踝轴前方 com_x_target。

    设计意图: 让维持姿态所需的踝力矩降到静摩擦底板(2 N·m)以下 —— 此时
    摩擦自己就担住了, 纯位置 PD 不需要为了出力而偏离指令角, base 俯仰误差
    也就不产生。这比"把质心放在支撑区几何中心"重要: 后者几何余量大但需 3+ N·m,
    会引入 5° 位置误差并被正反馈放大 2x。

    躯干垂直(base pitch=0) + 脚底贴平 ⇒ q1 + q4 + q5 = 0。以膝角 q4 为深度
    参数, 则只剩一个自由度, 由 CoM_x = 目标 定住 —— 逐深度一维求根。
    """
    def q_of(q1, q4):
        q = np.zeros(12)
        for off in (0, 6):
            q[off + 0], q[off + 3], q[off + 4] = q1, q4, -(q1 + q4)
            q[off + 1] = roll if off == 0 else -roll
            q[off + 2] = yaw if off == 0 else -yaw
        return q

    rows = []
    for q4 in np.arange(0.2, 1.65, 0.1):
        q1 = _solve_q1(m, q_of, q4, com_x_target, -1.6, 0.4)
        if q1 is None:
            rows.append((q4, None, None, None, None))
            continue
        r = analyze(m, q_of(q1, q4))
        rows.append((q4, q1, -(q1 + q4), r, torque_view(m, r)))
    return rows


def lean_squat_family(m, com_x_target=0.020, q5=-ANKLE_PITCH_LIM, roll=0.0, yaw=0.0):
    """踝钉在软限位、躯干俯仰自由的深蹲族。

    躯干垂直时 q1+q4+q5=0 把膝角锁在 0.8 以内(踝±0.4), 蹲不深。放开躯干
    俯仰 tb 后: 脚底贴平 + 踝取死 -0.4 使 膝以下直至髋 的位型完全由 q4 定住,
    只剩躯干绕髋转动这一个自由度(q1 = -(tb + q4 + q5))用来携质心。

    质心对 tb 是正弦关系(幅度仅 m_base/M * 136mm ≈ 39mm), 不单调, 所以不能对 q1
    二分; 改为扫 tb 取最靠近直立的那个根。足够深时两个根会消失 —— 那就是
    踝±0.4 给出的真正深度上限。
    """
    def q_of(tb, q4):
        q = np.zeros(12)
        for off in (0, 6):
            q[off + 0], q[off + 3], q[off + 4] = -(tb + q4 + q5), q4, q5
            q[off + 1] = roll if off == 0 else -roll
            q[off + 2] = yaw if off == 0 else -yaw
        return q

    def err(tb, q4):
        return analyze(m, q_of(tb, q4))["com"][0] - com_x_target

    rows = []
    for q4 in np.arange(0.3, 1.65, 0.1):
        grid = np.radians(np.arange(-30.0, 70.1, 1.0))
        e = [err(tb, q4) for tb in grid]
        roots = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                 if e[i] * e[i + 1] <= 0]
        if not roots:
            rows.append((q4, None, None, None, None))
            continue
        lo, hi = min(roots, key=lambda br: abs(0.5 * (br[0] + br[1])))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if err(lo, q4) * err(mid, q4) <= 0:
                hi = mid
            else:
                lo = mid
        tb = 0.5 * (lo + hi)
        r = analyze(m, q_of(tb, q4))
        rows.append((q4, -(tb + q4 + q5), q5, r, torque_view(m, r)))
    return rows


def _family_table(title, note, rows):
    print(f"\n\n=== {title} ===")
    print(note)
    print(f"{'膝 q4':>7}{'髋pitch q1':>11}{'踝pitch q5':>11}{'躯干前倾°':>10}"
          f"{'髋高mm':>8}{'质心高mm':>9}{'边距mm':>8}{'τ/只':>7}{'增益':>7}  限位")
    for q4, q1, q5, r, t in rows:
        if q1 is None:
            print(f"{q4:>7.2f}   无解(该深度下质心无法放到目标位置)")
            continue
        bad = []
        if abs(q5) > ANKLE_PITCH_LIM + 1e-9:
            bad.append(f"踝超{abs(q5)-ANKLE_PITCH_LIM:+.3f}")
        if q4 > 1.92:
            bad.append("膝超限")
        if q1 < -3.14:
            bad.append("髋超限")
        print(f"{q4:>7.2f}{q1:>11.3f}{q5:>11.3f}{r['pitch']:>10.1f}"
              f"{r['hip_h']*1000:>8.0f}{r['com'][2]*1000:>9.0f}"
              f"{r['margin']*1000:>8.1f}{t['tau_each']:>7.2f}{t['gain']:>7.3f}"
              f"  {'!! ' + '/'.join(bad) if bad else 'OK'}")


def report_family(m):
    _family_table(
        "质心中立下蹲族 A: 躯干垂直(q1+q4+q5=0)",
        "目标: 维持力矩 < 静摩擦底板 2 N·m, 让踝不必为出力而偏离指令角",
        com_neutral_family(m))
    _family_table(
        f"质心中立下蹲族 B: 踝钉在 -{ANKLE_PITCH_LIM:.1f}、躯干俯仰自由",
        "族 A 蹲不下去时的出路: 多余的屈膝角度由躯干前倾吃掉",
        lean_squat_family(m))


def com_authority(m, q_real, du=0.05):
    """质心水平权限: 各种关节联动模式每 0.1 rad 能把质心挪多远。

    踝被静摩擦锁在死区里(需求力矩 < 2 N·m 永远推不动它), 所以 IMU 外环只能
    靠髋/膝 把质心挪回去。这里算的就是那个反馈增益 dCoM_x/du。

    "膝+髋反向"是关键模式: 躯干指令俯仰 tb = -(q1+q4+q5) 不变, 仅把髋前后平移。
    "髋单独"只能转动上体(只占 28.7% 质量), 权限小一个数量级 —— 不能用它做主手。
    """
    modes = {
        "膝+髋反向 Δq4=+u,Δq1=-u": {3: +1.0, 0: -1.0},
        "髋单独   Δq1=+u": {0: +1.0},
        "膝单独   Δq4=+u": {3: +1.0},
    }
    out = {}
    for name, dq in modes.items():
        vals = []
        for sgn in (+1, -1):
            q = np.array(q_real, dtype=float)
            for off in (0, 6):
                for i, c in dq.items():
                    q[off + i] += sgn * c * du
            vals.append(analyze(m, q))
        out[name] = ((vals[0]["com"][0] - vals[1]["com"][0]) / (2 * du),
                     (vals[0]["com"][2] - vals[1]["com"][2]) / (2 * du))
    return out


def report_authority(m, poses):
    print("\n\n=== 质心水平权限(IMU 外环的可用增益) ===")
    print("踝不可用(需求力矩沉在 2~3 N·m 静摩擦死区里), 只能靠髋/膝挪质心")
    for nm, q in poses:
        print(f"  [{nm}]")
        for mode, (dx, dz) in com_authority(m, q).items():
            print(f"    {mode:<24} 质心x {dx*100:+7.1f} mm / 0.1rad"
                  f"   (附带质心高变化 {dz*100:+6.1f} mm)")


def build_waypoints(q0, goal, rounds=3, lean=0.30, lim=None):
    """STAND -> 深蹲 的路点序列: 先降低, 再交替卸载蹭开双脚, 最后归中顶挡块。

    为何不能直接张开: 终态两脚间距比 STAND 大 186 mm、每只脚还要原地扭 40°,
    而最终张角仅 12°(tan=0.22) 远小于 μ≈0.65 —— 全程摩擦占优, 重力不会帮忙。
    承重下蹭一只脚需 Fn(μ-tanθ)·力臂 ≈ 17.6 N·m, 对应 kp=100 下 0.176 rad 误差,
    超过看门狗阈值 0.156 -> 必跳闸。先侧倾把该脚卸到 ~20% 载荷, 需求降到
    3.5 N·m / 0.035 rad。

    侧倾 = 两腿髋roll 同向 + 踝roll 反向抵消(保持脚底贴平), 所以侧倾幅度同时被
    **踝roll 剩余行程**和**内收侧髋roll 只剩 0.26 rad**(腿撞机身)卡住。后者在
    第一轮最紧 —— 那条腿还没开始外展, 一点余量都没有。

    返回 [(名字, 12 维目标)], 调用方在相邻路点之间做 smoothstep 插值。
    """
    lo, hi = lim if lim is not None else real_limits()

    def cap(u, q):
        """把侧倾量夹到踝roll 与髋roll 两者都还剩得下的行程里。"""
        if u >= 0:
            room = [q[i] - lo[i] for i in LEAN_ANK] + [hi[i] - q[i] for i in LEAN_HIP]
        else:
            room = [hi[i] - q[i] for i in LEAN_ANK] + [q[i] - lo[i] for i in LEAN_HIP]
        return np.sign(u) * min(abs(u), max(0.0, min(room)))

    def leaned(q, u):
        q = q.copy()
        for i in LEAN_HIP:
            q[i] += u
        for i in LEAN_ANK:
            q[i] -= u
        return q

    q0, goal = np.asarray(q0, float), np.asarray(goal, float)
    wps = [("start 走路准备", q0.copy())]
    base = q0.copy()
    base[SEG_LIFT] = goal[SEG_LIFT]
    wps.append(("lift 降低(脚不动)", base.copy()))
    for i in range(1, rounds + 1):
        f = i / rounds
        for tag, idx, sgn in (("L", SPLAY_L, +1.0), ("R", SPLAY_R, -1.0)):
            other = "左" if sgn > 0 else "右"
            u = cap(sgn * lean, base)
            wps.append((f"lean{tag} 侧倾{np.degrees(abs(u)):.0f}°卸{other}脚", leaned(base, u)))
            base = base.copy()
            base[idx] = q0[idx] + f * (goal[idx] - q0[idx])
            u = cap(sgn * lean, base)
            wps.append((f"scrub{tag} 蹭{other}脚到 {f*100:.0f}%", leaned(base, u)))
    wps.append(("settle 归中顶挡块", goal.copy()))
    return wps


def report_waypoints(m, goal, rounds=3, lean=0.30):
    lo, hi = real_limits()
    wps = build_waypoints(STAND, goal, rounds, lean, (lo, hi))
    print(f"\n\n=== 蹭脚式轨迹校验({rounds} 轮, 侧倾上限 {np.degrees(lean):.0f}°, 限位=实机可达) ===")
    print(f"{'路点':<26}{'边距mm':>8}{'质心x':>7}{'质心y':>7}"
          f"{'间距mm':>8}{'偏航L':>7}{'偏航R':>7}  越限关节")
    worst = None
    for nm, q in wps:
        r = analyze(m, q, nm)
        qm = {f"joint_{n}": float(v) for n, v in zip(REAL, q)}
        T = m.fk(qm)
        zs = [T[f][0][:, 2] for f in ("Link_L6", "Link_R6")]
        Rb = align_rot(np.mean(zs, axis=0), [0, 0, 1])
        ft = {}
        for f in ("Link_L6", "Link_R6"):
            Rf, pf = T[f]
            RR = Rb @ Rf
            ft[f] = ((Rb @ pf)[1], np.degrees(np.arctan2(RR[1, 0], RR[0, 0])))
        over = []
        for j, n in enumerate(REAL):
            if not (lo[j] - 1e-6 <= q[j] <= hi[j] + 1e-6):
                over.append(f"{n}{q[j]:+.2f}")
        print(f"{nm:<26}{r['margin']*1000:>8.1f}{r['com'][0]*1000:>7.1f}"
              f"{r['com'][1]*1000:>7.1f}{(ft['Link_L6'][0]-ft['Link_R6'][0])*1000:>8.1f}"
              f"{ft['Link_L6'][1]:>7.1f}{ft['Link_R6'][1]:>7.1f}  "
              f"{','.join(over) if over else 'OK'}")
        if worst is None or r["margin"] < worst[1]:
            worst = (nm, r["margin"])
    print(f"\n最差路点: {worst[0]}  边距 {worst[1]*1000:+.1f} mm")


def main():
    m = Model(XML)
    lo, hi = real_limits()
    print(f"模型: {os.path.relpath(XML, os.path.dirname(HERE))}")
    print(f"可达行程: {os.path.relpath(COMMON_YAML, HERE)}"
          f"  (以实机能到的角度为准, 下面所有目标角与越限判据都读它)")
    print(f"整机质量 {m.mass_total:.3f} kg   mg = {m.mass_total*G:.1f} N")
    base_m = m.bodies["base"]["mass"]
    print(f"base(上体) {base_m:.3f} kg = 总质量 {base_m/m.mass_total*100:.1f}%,"
          f" 质心在 base 系 z={m.bodies['base']['com'][2]*1000:+.0f} mm")

    with open(BALANCE_JSON) as f:
        bal = np.array(json.load(f)["balance_pose_real_order"])
    # 录的是实机读数: 挡块承力时电机就停在挡块上, 读数会略微顶过软限位。
    # 按实机可达行程夹一道 -> 这才是真能下发的目标(差这一点不影响断电后的自锁)。
    off = [(REAL[i], bal[i]) for i in range(12)
           if not lo[i] - 1e-9 <= bal[i] <= hi[i] + 1e-9]
    bal = np.clip(bal, lo, hi)
    if off:
        print("录制姿态夹到可达行程: "
              + ", ".join(f"{n}{v:+.4f}" for n, v in off))

    for nm, q in (("走路准备姿态 STAND", STAND), ("下蹲平衡姿态 BALANCE", bal)):
        show(analyze(m, q, nm), m)

    # 沿 pose_move 的两段式插值(to_balance 顺序: 先 lift 后 shape)扫稳定边距
    print("\n\n=== 沿当前 to_balance 轨迹扫描(两段式: 先 lift 后 shape) ===")
    print(f"{'阶段':<8}{'s':>5}{'边距mm':>9}{'质心x':>8}{'质心y':>8}"
          f"{'pitch°':>9}{'髋高mm':>8}")
    worst = None
    for seg, idx in (("lift", SEG_LIFT), ("shape", SEG_SHAPE)):
        for s in np.linspace(0, 1, 11):
            q = STAND.copy()
            if seg == "shape":                       # lift 已完成
                q[SEG_LIFT] = bal[SEG_LIFT]
            q[idx] = STAND[idx] + s * (bal[idx] - STAND[idx])
            r = analyze(m, q, f"{seg} s={s:.1f}")
            print(f"{seg:<8}{s:>5.1f}{r['margin']*1000:>9.1f}"
                  f"{r['com'][0]*1000:>8.1f}{r['com'][1]*1000:>8.1f}"
                  f"{r['pitch']:>9.2f}{r['hip_h']*1000:>8.0f}")
            if worst is None or r["margin"] < worst["margin"]:
                worst = r
    print(f"\n最差点: {worst['label']}  边距 {worst['margin']*1000:+.1f} mm")
    if worst["margin"] < 0 and worst["normal"] is not None:
        d = worst["normal"]
        print(f"  越界方向 (x,y)=({d[0]:+.2f},{d[1]:+.2f})"
              f"  -> {'前' if d[0] > 0 else '后'}{'/左' if d[1] > 0 else '/右'}倾")

    report_waypoints(m, bal)

    report_family(m)

    sq = np.zeros(12)
    for off in (0, 6):
        sq[off + 0], sq[off + 3], sq[off + 4] = -0.867, 0.9, -0.4
    show(analyze(m, sq, "候选对称下蹲姿态(膝0.9/髋-0.867/踝-0.4)"), m)
    report_authority(m, [("走路准备 STAND", STAND), ("候选对称下蹲", sq)])


if __name__ == "__main__":
    main()
