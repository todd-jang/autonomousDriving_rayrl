"""Ray RLlib PPO smoke test — PerceptionEnv 5 iterations."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PerceptionEnv(gym.Env):
    def __init__(self, env_config=None):
        self.observation_space = spaces.Box(-1.,1.,shape=(13,),dtype=np.float32)
        self.action_space      = spaces.Box(-1.,1.,shape=(2,), dtype=np.float32)
        self._step = 0
    def reset(self,seed=None,options=None):
        self._step=0; return np.zeros(13,np.float32),{}
    def step(self,action):
        self._step+=1
        obs = np.clip(np.random.normal(0,.1,13).astype(np.float32),-1,1)
        rew = float(1.0 - abs(action[1])*0.5 - abs(obs[2])*1.5)
        return obs,rew,self._step>=200,False,{}

if __name__=="__main__":
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    gym.register("AutoDrive-v0",entry_point="rllib.smoke_test:PerceptionEnv",max_episode_steps=200)
    ray.init(ignore_reinit_error=True,num_cpus=2,log_to_driver=False,include_dashboard=False)
    algo=(PPOConfig()
        .environment("AutoDrive-v0")
        .framework("torch")
        .env_runners(num_env_runners=1,rollout_fragment_length=50)
        .training(train_batch_size=100,num_epochs=2)
        .resources(num_gpus=0)
        .build())
    print("\n=== RLlib Smoke Test ===")
    rewards=[]
    for i in range(5):
        r=algo.train()
        rw=r.get("env_runners",{}).get("episode_reward_mean",float("nan"))
        rewards.append(rw); print(f"  iter {i+1}/5 reward={rw:.3f}")
    algo.stop(); ray.shutdown()
    trend = rewards[-1]>rewards[0]-5.0 if len(rewards)>=2 else True
    print(f"\n{'✅ PASS' if trend else '❌ FAIL'} smoke test")
