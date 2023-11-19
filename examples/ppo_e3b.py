from rllte.agent import PPO
from rllte.env import make_minigrid_env
from rllte.xplore.reward import E3B

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_minigrid_env(
        device=device,
        num_envs=8,
        env_id="MiniGrid-Empty-6x6-v0",
    )
    
    # create agent
    agent = PPO(
        env=env, 
        device=device,
        tag="ppo_atari"
    )
    
    # create intrinsic reward
    e3b = E3B(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        num_envs=8,
        kappa=0,
        beta=1
    )
    
    print("==== AGENT ====")
    print(agent.encoder)
    print(agent.policy)
    
    print("==== E3B ====")
    print(e3b.elliptical_encoder)
    print(e3b.im)
    
    # set the module
    #agent.set(reward=e3b)
    
    # start training
    agent.train(num_train_steps=1_000_000)
