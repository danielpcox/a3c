import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
#rom torch.autograd import Variable
from torchvision import datasets, transforms
import time
from collections import deque


def test(rank, args, shared_model):

    if args.use_matplotlib:
        #import matplotlib and set backend
        import matplotlib
        matplotlib.use('macosx')
        import matplotlib.pyplot as plt

        #list of times and rewards for matplotlib
        rewards = []
        times = []
        to_save = 0

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        if args.show_game:
            env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.data
            hx = hx.data
        value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim = 1)
        action = prob.max(1)[1].data.numpy()

        state, reward, done, _ = env.step(action.item())
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action.item())
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))

            #plot times and rewards
            if args.use_matplotlib:
                times.append(time.gmtime(time.time() - start_time))
                rewards.append(reward_sum)
                if (to_save % 5 == 0):
                    print("saving graph...")
                    plt.plot(times, rewards)
                    plt.savefig("images/a3c.png")
                to_save+=1

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
