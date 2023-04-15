# Huggingface Diffusers Sprint 2023 - SUNDAE
> Project Code for Huggingface Diffusers Sprint 2023 on "Text Conditioned Step-unrolled Denoising Autoencoders are fast and controllable image generators."

**Team Members**
- [Alex McKinney](https://github.com/vvvm23)
- [Khalid Saifullah](https://github.com/khalidsaifullaah)
- [Christopher Galliart](https://github.com/HatmanStack)
- [Louis Wendler](https://github.com/1ucky40nc3)

## Repo Map
- `sundae/` - 2D Hourglass Transformer Code 
- `train-unconditional.py` - Unconditional model training
- `train_utils.py` - Generic training utils
- `utils.py` - Even more generic utils

## Setup
### Installation for development
Clone the repository:
```bash
git clone https://github.com/vvvm23/diffusers-sprint-sundae/git
cd diffusers-sprint-sundae
```
Install the requirements:
```
pip install -r requirements_dev.txt
pip install -e .
```

`TODO: setup instructions`
`TODO: add requirements file.`
`TODO: integration instructions`

## Usage
You can start a training run with the following command:
```bash
python train-unconditional.py \
    --output_dir ./outputs \
    --config ./sundae/configs/default.py
```
Our training script `train-uncoditional.py` has two required flags:
- `--output_dir`: A output directory for logs and checkpoints.
- `--config`: A config file in the in the [*ml_collections*](https://github.com/google/ml_collections) `ml_collections.config_flags` format.

Further details on how to work with `ml_collections.config_flags` can be found [here](https://github.com/google/ml_collections#:~:text=config_dict_initialization.py.-,Config%20Flags,-This%20library%20adds).


## Configuration
`TODO: instructions on configuring experiments`

### Acknowledgments
`TODO: add cool things that helped us :)`
