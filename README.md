# Paraphernalia

An assortment of tools for making digital art from Joe Halliwell
(@joehalliwell).

This is an incubator for immature scripts/modules with an absolute holy hell of
dependencies. Mature scripts/modules will be packaged more carefully and live
elsewhere. If/when I get around to it.

## Quick start guide

In a notebook:

```
!pip install --upgrade git+https://github.com/joehalliwell/paraphernalia.git
import paraphernalia as pa
pa.setup()
```

For developers: `poetry install`

## Features

- Fragment shader realtime preview and offline rendering
- CLIP-based image generation
- Helpers for running creative projects in jupyter/Colaboratory

## Extra/optional dependencies

- openai: CLIP and DALL-E models
- taming: Taming Transformers models

## TODOs

### General

- Tests
- Documentation
- Judicious type hints
- Helper for filenames/paths/projects

### glsl

- Support all/more Book of Shaders uniforms
- Support all Shadertoy uniforms (see https://github.com/iY0Yi/ShaderBoy)
- Support buffers

### jupyter

- Detect if running in Colaboratory and adjust accordingly

### torch

- CLIP: Adaptive focus
- Support batch restart properly
- Add BigGAN generators
- Fix crash when running tests on CPU
- Add soft_permutation()
- Work with target not latent space dims
- Factor our perceptual modes

### project

- Easy organization of project resources (inputs/outputs/checkpoints)
- Support for Colaboratory
