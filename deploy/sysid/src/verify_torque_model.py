#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""力矩层辨识/验证(走路数据, 与负载无关):
  τ_pd = kp(target-q) - kd*qd  (PD 命令, 未钳);  实机 τ_real 到关节帧 = kLegFbSign*tau
  - 时序: τ_real vs τ_pd + effort 线 -> 看实机在哪个力矩平顶(=真实 effort_limit)
  - 散点: (qd, τ_real) -> 边界是水平(effort钳位)还是斜线(转速滚降)
用法: python3 verify_torque_model.py <sim2real.csv> <out_dir>
"""
import sys, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# 实机序 idx0-11 = L1..L6,R1..R6
NM   = ["L1-HipPitch","L2-HipRoll","L3-HipYaw","L4-Knee","L5-AnkPitch","L6-AnkRoll",
        "R1-HipPitch","R2-HipRoll","R3-HipYaw","R4-Knee","R5-AnkPitch","R6-AnkRoll"]
KP   = [200,200,200,250,40,40]*2
KD   = [5,5,5,5,2,0.5]*2
EFF  = [26,26,26,26,26,5.8]*2
SIGN = [1,1,-1,1,-1,1, -1,1,-1,-1,1,1]        # kLegFbSign: tau(电机帧)->关节帧
TMAX = 30.0                                    # 达妙固件上限

def read(p):
    r=list(csv.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in r[0]}

def main():
    csv_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    R=read(csv_path); t=R["t"]-R["t"][0]
    JS=[1,7,4,10]                              # J2 L/R, J5 L/R
    fig,ax=plt.subplots(len(JS),2,figsize=(19,4.3*len(JS)))
    print(f"{'joint':<13}{'kp':>4}{'kd':>4}{'tau_pd p99':>11}{'tau_real p99':>13}{'corr(clip30)':>13}{'%pd>26':>8}{'%pd>30':>8}")
    for row,J in enumerate(JS):
        kp,kd,eff,sg=KP[J],KD[J],EFF[J],SIGN[J]
        cmd=R[f"cmd{J}"]; q=R[f"q{J}"]; qd=R[f"mvel{J}"]
        tau_real=sg*R[f"tau{J}"]               # -> 关节帧
        tau_pd=kp*(cmd-q)-kd*qd                # PD 命令(未钳)
        tau_c30=np.clip(tau_pd,-TMAX,TMAX)     # 模型: 钳到 30
        corr=np.corrcoef(tau_c30,tau_real)[0,1]
        p99pd=np.percentile(np.abs(tau_pd),99); p99r=np.percentile(np.abs(tau_real),99)
        f26=np.mean(np.abs(tau_pd)>26)*100; f30=np.mean(np.abs(tau_pd)>30)*100
        print(f"{NM[J]:<13}{kp:>4}{kd:>4}{p99pd:>11.1f}{p99r:>13.1f}{corr:>13.3f}{f26:>7.0f}%{f30:>7.0f}%")
        # 左: 时序 10-20s, τ_real vs τ_pd
        a=ax[row,0]; m=(t>=10)&(t<20)
        a.plot(t[m],tau_pd[m],color="tab:green",lw=1.0,alpha=0.8,label="τ_pd = kp·e − kd·qd (未钳)")
        a.plot(t[m],tau_real[m],color="tab:blue",lw=1.2,label="τ_real (关节帧)")
        for s in (1,-1):
            a.axhline(s*26,color="orange",ls="--",lw=1.0,label="effort 26" if s>0 else None)
            a.axhline(s*30,color="red",ls=":",lw=1.2,label="TMAX 30" if s>0 else None)
        a.set_title(f"{NM[J]}  [10-20s]  τ_real 平顶≈{p99r:.0f}  corr(model30)={corr:.2f}",fontsize=10)
        a.set_ylabel("torque [N·m]"); a.grid(alpha=0.3)
        if row==0: a.legend(fontsize=8,loc="upper right",ncol=2)
        # 右: (速度, 力矩) 散点
        b=ax[row,1]
        b.scatter(qd,tau_real,s=3,alpha=0.25,color="tab:blue")
        for s in (1,-1):
            b.axhline(s*26,color="orange",ls="--",lw=1.0); b.axhline(s*30,color="red",ls=":",lw=1.2)
        b.set_title(f"{NM[J]}  (velocity, torque) 散点",fontsize=10)
        b.set_xlabel("joint vel [rad/s]"); b.set_ylabel("torque [N·m]"); b.grid(alpha=0.3)
    ax[-1,0].set_xlabel("time [s]")
    fig.suptitle("Torque-level check (walking data): τ_real vs PD-model, and velocity-torque envelope",fontsize=13)
    fig.tight_layout()
    out=f"{out_dir}/torque_model_verify.png"; fig.savefig(out,dpi=115); plt.close(fig)
    print("saved",out)

if __name__=="__main__": main()
