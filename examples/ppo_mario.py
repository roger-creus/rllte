from rllte.agent import PPO
from rllte.env import make_mario_env


if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_mario_env(
        device=device,
        num_envs=16,
        env_id="SuperMarioBros-1-1-v3"
    )
    
    # create agent
    agent = PPO(
        env=env, 
        device=device,
        tag="ppo_mario",
        encoder_model="pathak",
        feature_dim=512,
        hidden_dim=512,
    )
    
    print("==== AGENT ====")
    print(agent.encoder)
    print(agent.policy)
    
    # start training
    agent.train(num_train_steps=10_000_000)
