# Paraphernalia

[![CI Badge](https://github.com/joehalliwell/paraphernalia/actions/workflows/test.yml/badge.svg)](https://github.com/joehalliwell/paraphernalia/actions)
[![CI Badge](https://github.com/joehalliwell/paraphernalia/actions/workflows/docs.yml/badge.svg)](https://github.com/joehalliwell/paraphernalia/actions)

An assortment of tools for making digital art from Joe Halliwell
(@joehalliwell).

## Features

- [Decent documentation](http://joehalliwell.com/paraphernalia)
- Fragment shader realtime preview and offline rendering
- CLIP-based image generation
- Helpers for running creative projects in jupyter/Colaboratory

## Quick start guide

In a notebook/Colaboratory:

```
!pip install --upgrade git+https://github.com/joehalliwell/paraphernalia.git[openai,taming]
import paraphernalia as pa
pa.setup()
```

For developers: `poetry install`

## TODOs

### General

- Add CLIP/generator sample notebook

### glsl

- Support all/more Book of Shaders uniforms
- Support all Shadertoy uniforms (see https://github.com/iY0Yi/ShaderBoy)
- Support buffers

### project

- Easy organization of project resources (inputs/outputs/checkpoints)

### torch

- Fix replace_grad and add tests
- Fix clamp_with_grad and add tests
- Add BigGAN generators
- Add soft_permutation()
- Add ZX Spectrum style generator
- Main entry point for generator+CLIP?
- Add standard description string/slug to generators

#### clip

- Is anti-prompt logic actually working?
- Adaptive focus
- Factor our perceptual modes
- Perceptual masking for CLIP
- Image prompts
- Add SRCNN
