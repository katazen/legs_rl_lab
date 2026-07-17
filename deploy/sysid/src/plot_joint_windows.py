#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单关节走路跟踪, 按 0-10/10-20/20-30s 分 3 子图放大。target/real/sim 三线。
用法: python3 plot_joint_windows.py <real.csv> <sim.csv> <out_dir> <joint_idx,...>
"""
import sys, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

JN = ["L1-HipPitch","L2-HipRoll","L3-HipYaw","L4-Knee","L5-AnkPitch","L6-AnkRoll",
      "R1-HipPitch","R2-HipRoll","R3-HipYaw","R4-Knee","R5-AnkPitch","R6-AnkRoll"]

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
        fig,ax=plt.subplots(3,1,figsize=(19,11))
        for w,(a,b) in enumerate(wins):
            mr=(tr>=a)&(tr<b); ms=(ts>=a)&(ts<b)
            ax[w].plot(tr[mr],R[f"cmd{J}"][mr],"k--",lw=1.6,label="target (real cmd)",zorder=5)
            ax[w].plot(tr[mr],R[f"q{J}"][mr],"-",color="tab:blue",lw=1.4,label="real (on ground)")
            ax[w].plot(ts[ms],S[f"q{J}"][ms],"-",color="tab:red",lw=1.4,label="sim (hung, legs.py cfg)")
            ax[w].set_title(f"{JN[J]}  [{a}-{b}s]",fontsize=12)
            ax[w].set_ylabel("pos [rad]"); ax[w].grid(alpha=0.3)
            if w==0: ax[w].legend(fontsize=10,loc="upper right")
        ax[-1].set_xlabel("time [s]")
        fig.suptitle(f"{JN[J]} walking tracking (real on ground vs sim hung)",fontsize=14)
        fig.tight_layout()
        out=f"{outdir}/track_{JN[J]}.png"; fig.savefig(out,dpi=115); plt.close(fig)
        print("saved",out)

if __name__=="__main__": main()
