# A3C

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

It is a revert, fork and update of [https://github.com/tpbarron/pytorch-a2c](https://github.com/tpbarron/pytorch-a2c) for teaching A3C.

There are still many outdated and deprecated PyTorch usages, and I'll complete the update later. I'm also planning to clean up after myself, since removing the `universe` dependency left a huge mess in envs.py. Because this is a teaching tool, I'll also be making an effort to clean up and simplify. All later.

For now, plays a near-perfect game of Pong after about 30 minutes of training on my MBP.

## Usage
```
python main.py --env-name "PongDeterministic-v4" --num-processes 16
```
