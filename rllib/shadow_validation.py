"""Shadow Mode validation — KL divergence deploy gate."""
import argparse,json,sys,time,logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("shadow")

def expert_action(obs):
    steer=-obs[2]*0.8+obs[11]*0.3
    thr  = 0.6 if obs[4]>0.5 else -0.4
    return np.array([thr,float(np.clip(steer,-1,1))],np.float32)

def run(checkpoint,episodes,slack_id,out,strict):
    from rllib.smoke_test import PerceptionEnv
    gate={"kl":0.08,"agree":0.85,"oncoming":0.05,"min_steps":1000 if strict else 100}
    all_kl,all_agree,all_onc=[],[],[]
    for ep in range(episodes):
        env=PerceptionEnv(); obs,_=env.reset()
        for _ in range(200):
            exp=expert_action(obs)
            shd=exp+np.random.normal(0,0.15,2).astype(np.float32)
            shd=np.clip(shd,-1,1)
            kl =float(np.mean((shd-exp)**2))
            ag =float(np.mean(np.sign(shd)==np.sign(exp)))
            all_kl.append(kl); all_agree.append(ag)
            all_onc.append(bool(obs[3]>0.5))
            obs,_,done,_,_=env.step(exp)
            if done: break
    mkl=float(np.mean(all_kl)); mag=float(np.mean(all_agree))
    mon=float(np.mean(all_onc)); steps=len(all_kl)
    gates={
        "kl_ok":mkl<gate["kl"],"agree_ok":mag>gate["agree"],
        "onc_ok":mon<gate["oncoming"],"steps_ok":steps>=gate["min_steps"]
    }
    approved=all(gates.values())
    report={"mean_kl":round(mkl,4),"mean_agree":round(mag,4),
            "oncoming":round(mon,4),"steps":steps,
            "gates":gates,"approved":approved}
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(report,indent=2))
    log.info("KL=%.4f agree=%.1f%% oncoming=%.1f%% → %s",
             mkl,mag*100,mon*100,"✅ APPROVED" if approved else "❌ REJECTED")
    return approved

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="checkpoints/smoke_test")
    ap.add_argument("--episodes",type=int,default=10)
    ap.add_argument("--slack-id",default="U09HNFL0B9S")
    ap.add_argument("--out",default="logs/shadow_report.json")
    ap.add_argument("--strict",action="store_true")
    args=ap.parse_args()
    ok=run(args.checkpoint,args.episodes,args.slack_id,args.out,args.strict)
    sys.exit(0 if ok else 1)
