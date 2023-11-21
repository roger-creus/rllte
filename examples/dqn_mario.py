from rllte.agent import DQN
from rllte.env import make_mario_env
from rllte.xplore.distribution import Categorical


if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    env = make_mario_env(
        device=device,
        num_envs=16,
        env_id="SuperMarioBros-1-1-v3"
    )
    
    # create agent
    agent = DQN(
        env=env, 
        device=device,
        tag="dqn_mario",
        encoder_model="pathak",
        feature_dim=512,
        hidden_dim=512,
    )

    agent.set(distribution=Categorical())
    
    print("==== AGENT ====")
    print(agent.encoder)
    print(agent.policy)
    
    # start training
    agent.train(num_train_steps=10_000_000)
