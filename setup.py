from setuptools import find_packages, setup


setup(
    name="sundae",
    version="0.0.1",
    description=(
        "Project Code for Huggingface Diffusers Sprint 2023 on"
        " 'Text Conditioned Step-unrolled Denoising Autoencoders are fast and controllable image generators.'"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
)