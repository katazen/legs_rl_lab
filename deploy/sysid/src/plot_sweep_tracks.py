#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每个 (vlim配置 × 关节) 一张图, 3 时间子图(0-10/10-20/20-30). target/real/sim(该vlim)。
用法: python3 plot_sweep_tracks.py <real.csv> <sweep_dir> <out_dir>
"""
import sys, csv, os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

JOINTS = [(1,"L2-HipRoll"),(7,"R2-HipRoll"),(4,"L5-AnkPitch"),(10,"R5-AnkPitch")]

def read(p):
    r=list(csv.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in r[0]}

def gain(tt,tgt,tq,q):
    qi=np.interp(tt,tq,q); return (qi-qi.mean()).std()/((tgt-tgt.mean()).std()+1e-9)

def main():
    real_csv, sweep_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    R=read(real_csv); tr=R["t"]-R["t"][0]
    files=sorted(glob.glob(f"{sweep_dir}/SIM_sweep_vlim*.csv"),
                 key=lambda p:float(re.search(r"vlim([0-9.]+)\.csv",p).group(1)))
    wins=[(0,10),(10,20),(20,30)]
    for f in files:
        vl=re.search(r"vlim([0-9.]+)\.csv",f).group(1)
        S=read(f); ts=S["t"]-S["t"][0]
        for idx,nm in JOINTS:
            rg=gain(tr,R[f"cmd{idx}"],tr,R[f"q{idx}"]); sg=gain(tr,R[f"cmd{idx}"],ts,S[f"q{idx}"])
            fig,ax=plt.subplots(3,1,figsize=(19,11))
            for w,(a,b) in enumerate(wins):
                mr=(tr>=a)&(tr<b); ms=(ts>=a)&(ts<b)
                ax[w].plot(tr[mr],R[f"cmd{idx}"][mr],"k--",lw=1.6,label="target",zorder=5)
                ax[w].plot(tr[mr],R[f"q{idx}"][mr],"-",color="tab:blue",lw=1.4,label=f"real (gain {rg:.2f})")
                ax[w].plot(ts[ms],S[f"q{idx}"][ms],"-",color="tab:red",lw=1.4,label=f"sim DCMotor vlim={vl} (gain {sg:.2f})")
                ax[w].set_title(f"{nm}  vlim={vl}  [{a}-{b}s]",fontsize=12)
                ax[w].set_ylabel("pos [rad]"); ax[w].grid(alpha=0.3)
                if w==0: ax[w].legend(fontsize=10,loc="upper right")
            ax[-1].set_xlabel("time [s]")
            fig.suptitle(f"{nm} @ DCMotor vlim={vl} (hung)  real g{rg:.2f} vs sim g{sg:.2f}",fontsize=13)
            fig.tight_layout()
            out=f"{out_dir}/{nm}_vlim{vl}.png"; fig.savefig(out,dpi=110); plt.close(fig)
            print("saved",os.path.basename(out))

if __name__=="__main__": main()
