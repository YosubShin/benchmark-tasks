"""Scorer tests: each hazard and cap before and after a milestone, and the timed hold (pure torch; no simulator)."""
import sys,torch
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tubecutter_sim.scorer import CutScorer, CutStep
def run(seq, hold=20):
    s=CutScorer(1,hold); out=[]
    for d,dent,spi,off,clr in seq:
        b=lambda v: torch.tensor([bool(v)])
        s.update(CutStep(torch.tensor([d]),b(dent),b(spi),b(off),b(clr)))
        out.append(int(s.score[0]))
    return out
ok=[]
def chk(name,got,exp): print(f"{name}: {got} expected {exp} {'OK' if got==exp else 'FAIL'}"); ok.append(got==exp)
good=[(0.0,0,0,0,0)]*3+[(0.45,0,0,0,0)]*3+[(0.9,0,0,0,0)]*3+[(1.36,0,0,0,1)]*25
chk("clean run final",run(good)[-1],100)
chk("clean run after 19 clear steps",run(good)[27],75)
chk("dent before any milestone",run([(0.0,1,0,0,0)]+good)[-1],0)
chk("dent after 0.5",run(good[:9]+[(0.9,1,0,0,0)]+good[9:])[-1],0)
chk("dent after 1.0",run(good+[(1.36,1,0,0,1)])[-1],0)
chk("spiral before milestone",run([(0.0,0,1,0,0)]+good)[-1],50)
chk("spiral after 0.5",run(good[:9]+[(0.9,0,1,0,0)]+good[9:])[-1],50)
chk("off mark after 0.25",run(good[:4]+[(0.45,0,0,1,0)]+good[4:])[-1],50)
s=good[:9]+[(1.36,0,0,0,1)]*15+[(1.36,0,0,0,0)]+[(1.36,0,0,0,1)]*15
chk("hold broken at 15 of 20 steps",run(s)[-1],75)
chk("hold completes after the break",run(s+[(1.36,0,0,0,1)]*5)[-1],100)
print("ALL OK" if all(ok) else "SOME FAILED")
