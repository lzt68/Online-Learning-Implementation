# README

This folder aims to reproduce the result of "One group of suboptimal" setting in Karnin2013 Figure 1.

## File Structure

"Source": 

+ The source file of Sequential Halving Algorithm, the consumption environment and a  function used for conducting numeric simulation
  + class SequentialHalving_Agent: In each phase, adopt round robin to pull the arms
  + class SequentialHalving_Agent_Variant: In each phase, we pre-calculate the pulling times of each arm, and then pull them in a row

SeqHalving-SeqHalvingVariant.ipynb: Demo for how to use the source file.

## Demo

```python
from Source.source import *
import numpy as np
K = 10 # arm number
C = 100 # total budget, equivalent to T

success_rate, std_success_rate, pulling_times_, predict_arm_, best_arm_ = Experiment(
    reward=np.array([0.5] + [0.4] * 9),
    demand=np.ones(K), # mean consumption of each arm
    env_class=Env_FixedConsumption,
    env_para=dict(),
    agent_class=SequentialHalving_Agent,
    agent_para=dict(),
    n_experiment=100,
    K=K,
    C=C,
)
print(f"success rate is {success_rate}, the standard deviation is {std_success_rate}")
```

```python
from Source.source import *
import numpy as np
K = 10 # arm number
C = 100 # total budget, equivalent to T

success_rate, std_success_rate, pulling_times_, predict_arm_, best_arm_ = Experiment(
    reward=np.array([0.5] + [0.4] * 9),
    demand=np.ones(K), # mean consumption of each arm
    env_class=Env_FixedConsumption,
    env_para=dict(),
    agent_class=SequentialHalving_Agent_Variant,
    agent_para=dict(),
    n_experiment=100,
    K=K,
    C=C,
)
print(f"success rate is {success_rate}, the standard deviation is {std_success_rate}")
```

## Unsolved Problem

These two implementation of Sequential Halving Algorithm achieves similar performance, but I still cannot reproduce the performance mentioned in Karnin2013. The accuracy I got is smaller than their proposed result in nearly all the experiment settings.

