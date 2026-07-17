#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""辨识 J2/J5 vlim: 对比实机 gain 与各 vlim 的 sim gain, 找交点 = 辨识 vlim。
用法: python3 plot_vlim_sweep.py <real.csv> <sweep_dir> <out_dir>
"""
import sys, csv, os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# 关节: (实机idx, 名字)
JOINTS = [(1,"L2-HipRoll"),(7,"R2-HipRoll"),(4,"L5-AnkPitch"),(10,"R5-AnkPitch")]

def read(p):
    r=list(csv.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in r[0]}

def gain(tt, tgt, tq, q):
    qi=np.interp(tt,tq,q); g0=tgt-tgt.mean(); r0=qi-qi.mean()
    return r0.std()/(g0.std()+1e-9)

def main():
    real_csv, sweep_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    R=read(real_csv); tr=R["t"]-R["t"][0]
    files=sorted(glob.glob(f"{sweep_dir}/SIM_sweep_vlim*.csv"),
                 key=lambda p:float(re.search(r"vlim([0-9.]+)\.csv",p).group(1)))
    vlims=[float(re.search(r"vlim([0-9.]+)\.csv",p).group(1)) for p in files]
    sims=[read(p) for p in files]

    # real gain / sim gain(每 vlim)
    real_g={}; sim_g={}
    for idx,nm in JOINTS:
        real_g[nm]=gain(tr, R[f"cmd{idx}"], tr, R[f"q{idx}"])
        sim_g[nm]=[gain(tr, R[f"cmd{idx}"], S["t"]-S["t"][0], S[f"q{idx}"]) for S in sims]

    # 找交点(sim gain == real gain), 线性插值
    ident={}
    for idx,nm in JOINTS:
        sg=np.array(sim_g[nm]); rg=real_g[nm]; vv=np.array(vlims)
        d=sg-rg
        vfit=None
        for i in range(len(vv)-1):
            if d[i]==0: vfit=vv[i]
            elif d[i]*d[i+1]<0:
                vfit=vv[i]+(vv[i+1]-vv[i])*(0-d[i])/(d[i+1]-d[i]); break
        ident[nm]=vfit

    print(f"{'joint':<13}{'real gain':>10}   sim gain per vlim "+ " ".join(f"{v:g}" for v in vlims))
    for idx,nm in JOINTS:
        print(f"{nm:<13}{real_g[nm]:>10.2f}   "+" ".join(f"{g:.2f}" for g in sim_g[nm])
              +f"   -> vlim* = {ident[nm]}")

    # Fig: gain vs vlim
    fig,ax=plt.subplots(2,2,figsize=(15,9))
    for a,(idx,nm) in zip(ax.ravel(),JOINTS):
        a.plot(vlims, sim_g[nm], "o-", color="tab:red", label="sim gain")
        a.axhline(real_g[nm], color="tab:blue", ls="--", lw=1.5, label=f"real gain {real_g[nm]:.2f}")
        if ident[nm]: a.axvline(ident[nm], color="green", ls=":", lw=1.5, label=f"vlim* = {ident[nm]:.2f}")
        a.set_title(nm, fontsize=12); a.set_xlabel("velocity_limit [rad/s]"); a.set_ylabel("tracking gain")
        a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.suptitle("Identify vlim from walking: sim gain(vlim) vs real gain", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out_dir}/vlim_identify.png",dpi=120); plt.close(fig)
    print("saved", f"{out_dir}/vlim_identify.png")

    # Fig: tracking at nearest-swept vlim to identified (3 windows) per joint
    wins=[(0,10),(10,20),(20,30)]
    for idx,nm in JOINTS:
        if ident[nm] is None: continue
        bi=int(np.argmin(np.abs(np.array(vlims)-ident[nm]))); S=sims[bi]; ts=S["t"]-S["t"][0]
        fig,ax=plt.subplots(3,1,figsize=(19,11))
        for w,(a,b) in enumerate(wins):
            mr=(tr>=a)&(tr<b); ms=(ts>=a)&(ts<b)
            ax[w].plot(tr[mr],R[f"cmd{idx}"][mr],"k--",lw=1.6,label="target",zorder=5)
            ax[w].plot(tr[mr],R[f"q{idx}"][mr],"-",color="tab:blue",lw=1.4,label="real")
            ax[w].plot(ts[ms],S[f"q{idx}"][ms],"-",color="tab:red",lw=1.4,label=f"sim DCMotor vlim={vlims[bi]:g}")
            ax[w].set_title(f"{nm} [{a}-{b}s]",fontsize=12); ax[w].set_ylabel("pos [rad]"); ax[w].grid(alpha=0.3)
            if w==0: ax[w].legend(fontsize=10,loc="upper right")
        ax[-1].set_xlabel("time [s]")
        fig.suptitle(f"{nm} tracking at identified vlim≈{ident[nm]:.2f} (real vs sim DCMotor, hung)",fontsize=13)
        fig.tight_layout(); fig.savefig(f"{out_dir}/track_ident_{nm}.png",dpi=115); plt.close(fig)
        print("saved", f"{out_dir}/track_ident_{nm}.png")

if __name__=="__main__": main()
