Currently, Hsuanwu only supports online pre-training via intrinsic reward. To turn on the pre-training mode, 
it suffices to adjust the config file like:
``` yaml title="config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.

agent:
  name: DrQv2             # The agent name.
```
The adjusted config:
``` yaml title="new_config.yaml"
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...).
seed: 1                   # Random seed for reproduction.
num_train_steps: 5000     # Number of training steps.
pretraining: true         # turn on the pre-training mode.

agent:
  name: DrQv2             # The agent name.

####### Choose the intrinsic reward
reward:
 name: RE3                # The reward name.
 ...
```
Run the `train.py`, and you will see the following output:
``` sh
[04/30/2023 02:58:56 PM] - [HSUANWU INFO ] - Experiment: drqv2_dmc_pixel
[04/30/2023 02:58:56 PM] - [HSUANWU INFO ] - Invoking Hsuanwu Engine...
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Checking the Compatibility of Modules...
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Selected Encoder: TassaCnnEncoder
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Selected Agent: DrQv2
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Selected Storage: NStepReplayStorage
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Selected Distribution: TruncatedNormalNoise
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Use Augmentation: True, RandomShift
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Use Intrinsic Reward: True, RE3
[04/30/2023 02:58:56 PM] - [HSUANWU INFO ] - Deploying OffPolicyTrainer...
[04/30/2023 02:58:56 PM] - [HSUANWU INFO ] - Pre-training Mode On...
[04/30/2023 02:58:56 PM] - [HSUANWU DEBUG] - Check Accomplished. Start Training...
```

!!! tip
    When the pre-training mode is on, a `reward` module must be specified!
For all supported reward modules, see [API Documentation](https://docs.hsuanwu.dev/api/).