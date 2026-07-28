#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真·拟合: 对 chirp 的复频响拟合二阶+延迟模型 H=kp*e^{-jwτ}/(kp - J w² + j c w)。
最小二乘解 J(惯量), c(总阻尼=kd+粘滞), τ(纯延迟)。纯 numpy 网格+局部细化, 不依赖 scipy。
用法: python3 fit_actuator.py <chirp.csv> <kp> [fmin fmax]
"""
import sys, csv
import numpy as np

def frf(t, u, y, f0, f1, nbin=50):
    u=u-u.mean(); y=y-y.mean(); dt=np.median(np.diff(t)); n=len(t); win=np.hanning(n)
    U=np.fft.rfft(u*win); Y=np.fft.rfft(y*win); f=np.fft.rfftfreq(n,dt)
    edges=np.logspace(np.log10(f0),np.log10(f1),nbin+1); fc=[]; H=[]; W=[]
    for i in range(nbin):
        m=(f>=edges[i])&(f<edges[i+1])
        p=(np.abs(U[m])**2).sum()
        if m.sum()<1 or p<1e-10: continue
        fc.append(np.sqrt(edges[i]*edges[i+1]))
        H.append((Y[m]*np.conj(U[m])).sum()/p); W.append(p)   # H1 + 权重=输入能量
    return np.array(fc), np.array(H), np.array(W)

def model(f, kp, J, c, td):
    w=2*np.pi*f
    return kp*np.exp(-1j*w*td)/(kp - J*w**2 + 1j*c*w)

def fit(fc, H, W, kp):
    W = np.ones_like(W)   # 等权(log 扫频 |U|² 低频过大, 会淹没共振峰), 每频段同等重要
    Js=np.linspace(0.05,1.6,90); cs=np.linspace(2,35,90); tds=np.linspace(0,0.06,40)
    best=(1e18,None)
    for J in Js:
        for c in cs:
            # 对每个(J,c), τ 只影响相位, 扫 td 取最优
            for td in tds:
                Hm=model(fc,kp,J,c,td)
                e=np.sum(W*np.abs(H-Hm)**2)
                if e<best[0]: best=(e,(J,c,td))
    J,c,td=best[1]
    # 局部细化
    for _ in range(3):
        Js=np.linspace(J*0.8,J*1.2,25); cs=np.linspace(c*0.8,c*1.2,25); tds=np.linspace(max(0,td-0.01),td+0.01,25)
        best=(1e18,None)
        for JJ in Js:
            for cc in cs:
                for tt in tds:
                    e=np.sum(W*np.abs(H-model(fc,kp,JJ,cc,tt))**2)
                    if e<best[0]: best=(e,(JJ,cc,tt))
        J,c,td=best[1]
    return J,c,td,best[0]

def main():
    path=sys.argv[1]; kp=float(sys.argv[2])
    f0=float(sys.argv[3]) if len(sys.argv)>3 else 0.8
    f1=float(sys.argv[4]) if len(sys.argv)>4 else 8.0
    r=list(csv.DictReader(open(path)))
    t=np.array([float(x["t"]) for x in r]); tgt=np.array([float(x["target"]) for x in r]); q=np.array([float(x["q"]) for x in r])
    t-=t[0]
    fc,H,W=frf(t,tgt,q,f0,f1)
    J,c,td,e=fit(fc,H,W,kp)
    wn=np.sqrt(kp/J); zeta=c/(2*np.sqrt(kp*J))
    print(f"拟合 {path}  (band {f0}-{f1}Hz, kp={kp:g})")
    print(f"  J(等效惯量)   = {J:.4f} kg·m²")
    print(f"  c(总阻尼=kd+粘滞) = {c:.2f} N·m·s/rad")
    print(f"  τ(纯延迟)      = {td*1000:.1f} ms  ({td/0.005:.1f} 物理步@200Hz)")
    print(f"  -> 无阻尼固有频率 fn={wn/2/np.pi:.2f}Hz, 阻尼比 ζ={zeta:.3f}")

if __name__=="__main__": main()
