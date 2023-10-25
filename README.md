# Offline Bisimulation
Official pytorch implementation of [Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning]().

## Requirements

The exact requirements depend on the baseline methods, e.g., for applying bisimulation methods on TD3BC, one can follow the requirements from [official repo of TD3BC](https://github.com/sfujim/TD3_BC).

## Usage

Pretrain with baseline method - [SimSR](https://arxiv.org/abs/2112.15303): `python main.py --env hopper-medium-expert-v2 --obj simsr --slope 0.5 --reward_norm False --reward_scale False`

Pretrain with baseline method - [MiCO](https://arxiv.org/abs/2106.08229): `python main.py --env hopper-medium-expert-v2 --obj mico --slope 0.5 --reward_norm False --reward_scale False`

Pretrain with our method: `python main.py --env hopper-medium-expert-v2 --obj simsr --slope 0.7 --reward_norm True --reward_scale True`



## Citation
If you find this open source release useful, please reference it in your paper:
```
@inproceedings{
zang2023understanding,
title={Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning},
author={Hongyu Zang and Xin Li and Leiji Zhang and Yang Liu and Baigui Sun and Riashat Islam and Remi Tachet des Combes and Romain Laroche},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=sQyRQjun46}
}
```
