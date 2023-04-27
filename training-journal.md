## Sundae Text-to-Image Training Journal

I was inspired by Boris' training journal which I was reminded existed during his kick off talk for the event.
Here are my own rough notes of tuning parameters and the main training progress.
We don't have time for a super deep search of hyperparameters parameters, but let us do what we can, as we will only have one shot :)

### Where we are starting from
> 2023/04/22

Got nearly all the components together, just need the data loader and sweep infrastructure.

Boris mentioned one of the most important parameters to tune is learning rate but it doesn't translate well with scaling, so we should leave it last but it is still most important.

I'll be following (roughly) [this guide](https://github.com/google-research/tuning_playbook#choosing-a-model-architecture)

Currently practising sweeps on an unconditional model. I'll move to first stage of sweeps as soon as dataloading is there.

# First Stage of tuning
This was somewhat cut short initially as we noticed the training data was corrupted - wrong scaling.

I am focusing on a reduce learning rate search, and emphasizing exploring model dim and depth at high levels versus low levels.

Initial results from unconditional suggested parallel vs non parallel is about the same. Went with the latter.
Tied embeddings made training super unstable.
Also suggested more heads is good, fixing at 16.
We decided to fix shorten factor at 4 for efficiency :) same for unroll.

# Second and Third Stage of tuning
We found that the next stage of sweeps resulted in the top 6 results being perfectly balanced among all 6 possible architectures :( so wholy inconclusive. This did allow us to fix the learning rate at 1e-4. 

The final stage of sweeps fixed the learning rate here and tried two final models with depth (2, 24, 2) and (3, 16, 3). They were both the same, so we went with the former as it had the most parameters. Again, most our sweeps were not too conclusive which is unfortunate. The run also crashed so couldn't run for as many steps as we wanted.

