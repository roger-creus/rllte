import torch as th
import imageio
from rllte.env import make_mario_env
from torch.distributions import Categorical
import numpy as np

if __name__ == "__main__":
    # env setup
    device = "cpu"
    env = make_mario_env(
        device=device,
        num_envs=1,
        env_id="SuperMarioBros-1-1-v3",
        asynchronous=False
    )

    # load the model and specify the map location
    agent = th.load("/home/roger/Desktop/rllte/logs/ppo_mario/2023-11-20-09-46-19/model/agent_206848.pth", map_location=th.device('cpu'))

    # evaluate the agent and record video of 1 episode
    done = False
    obs, _ = env.reset()
    ep_frames = []
    ep_reward = 0
    ep_len = 0
    while not done:
        with th.no_grad():
            ep_frames.append(obs[0][-1].cpu().numpy())
            action = agent(obs / 255.)
            dist = Categorical(logits=action)
            action = dist.sample()
            obs, r, term, trunc, _ = env.step(action)
            done = np.logical_or(term.cpu().numpy(), trunc.cpu().numpy())
            ep_reward += r.cpu().numpy().item()
            print(action.cpu().numpy(), done)
            ep_len += 1

    print("Episode reward: ", ep_reward, "Episode length: ", ep_len)
    
    # save the video
    imageio.mimsave("video.gif", ep_frames, fps=30)