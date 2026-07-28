#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chirp -> Bode(增益/相位 vs 频率)。可叠加多条(sim/real)。纯 numpy(H1 分频段平均)。
用法: python3 bode.py <out.png> <f0> <f1> <csv1:label1> [csv2:label2 ...]
  csv 需含列 t,target,q
"""
import sys, csv as csvmod
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def read(p):
    r=list(csvmod.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in ["t","target","q"]}

def frf(t, u, y, f0, f1, nbin=60):
    """H1 = <Y·U*>/<U·U*>, 按对数频段平均。返回 fc, gain, phase(deg,解缠)。"""
    u=u-u.mean(); y=y-y.mean()
    dt=np.median(np.diff(t)); n=len(t)
    win=np.hanning(n)
    U=np.fft.rfft(u*win); Y=np.fft.rfft(y*win)
    f=np.fft.rfftfreq(n, dt)
    edges=np.logspace(np.log10(f0), np.log10(f1), nbin+1)
    fc, H = [], []
    for i in range(nbin):
        m=(f>=edges[i])&(f<edges[i+1])
        if m.sum()<1 or (np.abs(U[m])**2).sum()<1e-12: continue
        Huu=(Y[m]*np.conj(U[m])).sum()/(U[m]*np.conj(U[m])).sum()   # H1
        fc.append(np.sqrt(edges[i]*edges[i+1])); H.append(Huu)
    fc=np.array(fc); H=np.array(H)
    gain=np.abs(H); ph=np.degrees(np.unwrap(np.angle(H)))
    return fc, gain, ph

def main():
    out=sys.argv[1]; f0=float(sys.argv[2]); f1=float(sys.argv[3])
    items=[s.split(":") for s in sys.argv[4:]]
    fig,ax=plt.subplots(2,1,figsize=(12,9),sharex=True)
    cols=["tab:red","tab:blue","tab:green","tab:orange"]
    for k,(path,lab) in enumerate(items):
        d=read(path); fc,g,ph=frf(d["t"],d["target"],d["q"],f0,f1)
        c=cols[k%len(cols)]
        ax[0].semilogx(fc,g,"-o",ms=3,color=c,label=lab)
        ax[1].semilogx(fc,ph,"-o",ms=3,color=c,label=lab)
        # 特征: -3dB 带宽(gain 首次跌破 0.707*低频增益), 共振峰
        g0=g[:3].mean(); ipk=np.argmax(g);
        bw=None
        below=np.where(g<0.707*g0)[0]
        if len(below): bw=fc[below[0]]
        peak_txt = f"共振峰 {g[ipk]:.2f}@{fc[ipk]:.1f}Hz" if g[ipk]>1.05*g0 else "无明显共振峰"
        print(f"[{lab}] 低频增益{g0:.2f}  {peak_txt}  -3dB带宽~{bw:.1f}Hz" if bw else
              f"[{lab}] 低频增益{g0:.2f}  {peak_txt}  -3dB带宽>{f1}Hz")
        # 相位斜率估延迟(取 1-5Hz 段线性拟合 dphase/df -> tau=-slope/360)
        seg=(fc>=1)&(fc<=5)
        if seg.sum()>=2:
            slope=np.polyfit(fc[seg],ph[seg],1)[0]   # deg/Hz
            tau=-slope/360.0*1000                     # ms
            print(f"       相位@1-5Hz 斜率 {slope:.0f}°/Hz -> 纯延迟 ~{tau:.0f} ms")
    ax[0].axhline(0.707,color="gray",ls=":",lw=1,label="-3dB (0.707)"); ax[0].axhline(1.0,color="k",lw=0.5)
    ax[0].set_ylabel("gain |q/target|"); ax[0].grid(alpha=.3,which="both"); ax[0].legend(); ax[0].set_title("Bode: gain")
    ax[1].axhline(0,color="k",lw=0.5); ax[1].set_ylabel("phase [deg]"); ax[1].set_xlabel("frequency [Hz]")
    ax[1].grid(alpha=.3,which="both"); ax[1].legend(); ax[1].set_title("Bode: phase (lag)")
    fig.tight_layout(); fig.savefig(out,dpi=120); print("saved",out)

if __name__=="__main__": main()
