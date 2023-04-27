# Huggingface Diffusers Sprint 2023 - SUNDAE
> Project Code for Huggingface Diffusers Sprint 2023 on "Text Conditioned Step-unrolled Denoising Autoencoders are fast and controllable image generators."

**Team Members**
- [Alex McKinney](https://github.com/vvvm23)
    - Contributed SUNDAE training and inference code, porting Hourglass transformer.
- [Khalid Saifullah](https://github.com/khalidsaifullaah)
    - Contributed dataset creation and subsequent VQ-GAN processing.
- [Christopher Galliart](https://github.com/HatmanStack)
    - Contributed WandB Integration and Streamlit demo on Huggingface.
- [Louis Wendler](https://github.com/1ucky40nc3)
    - Contributed text-to-image dataloaders, OpenCLIP, Wandb Integration, and training utils.

## Repo Map
- `sundae/` - 2D Hourglass Transformer Code 
- `sundae/configs` – Configuration files for experiments
- `data_encoder.py` – VQ-GAN dataset processing script
- `dev/data_utils.py` – Dataset downloading utils
- `train_utils.py` - Generic training utils
- `sample_utils.py` - Optimised sampling utils
- `utils.py` - Even more generic utils
- `train_unconditional.py` - Unconditional model training
- `train_text_to_image.py` - Text-conditioned model training
- `inference_unconditional.py` – Inference script for unconditional model.
- `main.py` – Central script for launching experiments and loading configs.

## Setup
### Installation for development
Clone the repository:
```shell
git clone https://github.com/vvvm23/diffusers-sprint-sundae/git
cd diffusers-sprint-sundae
```
Install Jax for your system and specific accelerator(s), then install the requirements:
```shell
pip install -r requirements_dev.txt
pip install -e .
```

If you want to use `wandb` during training, please run:
```shell
wandb login
```

## Usage
You can start a training run with the following command:
```shell
python main.py --config ./sundae/configs/default.py
```
The config file should be in the in the [`ml_collections`](https://github.com/google/ml_collections) `ml_collections.config_flags` format.
Further details on how to work with `ml_collections.config_flags` can be found [here](https://github.com/google/ml_collections#:~:text=config_dict_initialization.py.-,Config%20Flags,-This%20library%20adds).

You can also start your training runs (for the moment) by calling the training scripts directly.
Just execute one of the following commands and you will be fine:
```shell
python train_unconditional.py
python train_text_to_image.py
```

To start a training run with conditional text-to-image training you can call the command below.
Just remember to substitute the `<enter-train-file-here>` string with the path to your training directory.
Notes: 
- It is advised to use training file in the ".parquet" format.
```bash
python main.py \
    --config ./sundae/configs/text_to_image.py
    --config.data.train_dir <enter-train-dir-here>
```

`TODO: add instructions for accessing inference via main.py`

`TODO: add instructions for text-to-image inference.`

`TODO: add instructions for launching Streamlit demo.`


## Configuration
We handle the command-line flags and configuration with abseil and ml_collections.
This allows for easy access to configuration from the command-line and inside Python modules.
instructions for the command-line usage can be found above.
Our config files in the [`configs`](sundae/configs) directory are Python modules that implement
a `get_config` method which returns a `ml_collections.ConfigDict`. Look into the [`default.py`](sundae/configs/default.py) config for guidance.

---

### Paper references
[Hourglass Transformers](https://arxiv.org/abs/2110.13711)

[SUNDAE](https://arxiv.org/abs/2112.06749)

[SUNDAE x VQGAN](https://arxiv.org/abs/2206.12351)


### Acknowledgments
- The Huggingface Team for their great transformers and diffusers examples
- [lucidrains](https://github.com/lucidrains) for his great Hourglass PyTorch implementation
- Google Cloud for generously providing free TPU hours.
