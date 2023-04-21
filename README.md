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
Install Jax for your system, then install the requirements:
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
python main.py \
    --output_dir ./outputs \
    --config ./sundae/configs/default.py
```
Our main script `main.py` has two required flags:
- `--output_dir`: A output directory for logs and checkpoints.
- `--config`: A config file in the in the [*ml_collections*](https://github.com/google/ml_collections) `ml_collections.config_flags` format.

Further details on how to work with `ml_collections.config_flags` can be found [here](https://github.com/google/ml_collections#:~:text=config_dict_initialization.py.-,Config%20Flags,-This%20library%20adds).

You can also start your training runs (for the moment) by calling the training scripts directly.
Just execute one of the following commands and you will be fine:
```bash
python train_unconditional.py
python train_text_to_image.py
```

## Configuration
We handle the command-line flags and configuration with abseil and ml_collections.
This allows for easy access to configuration from the command-line and inside Python modules.
instructions for the command-line usage can be found above.
Our config files in the [`configs`](sundae/configs) directory are Python modules that implement
a `get_config` method which returns a `ml_collections.ConfigDict`. Look into the [`default.py`](sundae/configs/default.py) config for guidance.

### References
[Hourglass Transformers](https://arxiv.org/abs/2110.13711)

[SUNDAE](https://arxiv.org/abs/2112.06749)

[SUNDAE x VQGAN](https://arxiv.org/abs/2206.12351)


### Acknowledgments
- The Huggingface Team for their great transformers and diffusers examples
- [lucidrains](https://github.com/lucidrains) for his great Hourglass PyTorch implementation
