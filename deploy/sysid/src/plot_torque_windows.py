#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单关节走路力矩, 按 0-10/10-20/20-30s 分 3 子图放大。标力矩上下限 + 实机(和sim)实时力矩。
用法: python3 plot_torque_windows.py <real.csv> <sim.csv> <out_dir> <joint_idx,...>
"""
import sys, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

JN = ["L1-HipPitch","L2-HipRoll","L3-HipYaw","L4-Knee","L5-AnkPitch","L6-AnkRoll",
      "R1-HipPitch","R2-HipRoll","R3-HipYaw","R4-Knee","R5-AnkPitch","R6-AnkRoll"]
EFF_SIM=[26,26,26,26,26,5.8]*2          # legs.py effort_limit_sim
TMAX_REAL=[30,30,30,30,30,30,30,30,30,30,30,30]  # 达妙固件 TMAX (实机可达上限)

def read(p):
    r=list(csv.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in r[0]}

def main():
    real_csv,sim_csv,outdir=sys.argv[1],sys.argv[2],sys.argv[3]
    joints=[int(x) for x in sys.argv[4].split(",")]
    os.makedirs(outdir,exist_ok=True)
    R=read(real_csv); S=read(sim_csv)
    tr=R["t"]-R["t"][0]; ts=S["t"]-S["t"][0]
    wins=[(0,10),(10,20),(20,30)]
    for J in joints:
        eff=EFF_SIM[J]; tmax=TMAX_REAL[J]
        fig,ax=plt.subplots(3,1,figsize=(19,11))
        for w,(a,b) in enumerate(wins):
            mr=(tr>=a)&(tr<b); ms=(ts>=a)&(ts<b)
            ax[w].plot(tr[mr],R[f"tau{J}"][mr],"-",color="tab:blue",lw=1.2,label="real torque")
            ax[w].plot(ts[ms],S[f"tau{J}"][ms],"-",color="tab:red",lw=0.9,alpha=0.7,label="sim torque")
            for s in (+1,-1):
                ax[w].axhline(s*eff,color="orange",ls="--",lw=1.2,label=("sim effort_limit %g"%eff) if (w==0 and s>0) else None)
                ax[w].axhline(s*tmax,color="red",ls=":",lw=1.2,label=("real TMAX %g"%tmax) if (w==0 and s>0) else None)
            frac=np.mean(np.abs(R[f"tau{J}"][mr])>0.9*tmax)*100
            ax[w].set_title(f"{JN[J]}  [{a}-{b}s]   real|tau|max={np.abs(R[f'tau{J}'][mr]).max():.1f}  near-TMAX {frac:.0f}%",fontsize=12)
            ax[w].set_ylabel("torque [N·m]"); ax[w].grid(alpha=0.3)
            if w==0: ax[w].legend(fontsize=9,loc="upper right",ncol=2)
        ax[-1].set_xlabel("time [s]")
        fig.suptitle(f"{JN[J]} walking torque (real vs sim, limits marked)",fontsize=14)
        fig.tight_layout()
        out=f"{outdir}/torque_{JN[J]}.png"; fig.savefig(out,dpi=115); plt.close(fig)
        print("saved",out)

if __name__=="__main__": main()
