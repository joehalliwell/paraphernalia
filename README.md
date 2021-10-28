# Paraphernalia

[![CI Badge](https://github.com/joehalliwell/paraphernalia/actions/workflows/test.yml/badge.svg)](https://github.com/joehalliwell/paraphernalia/actions)
[![CI Badge](https://github.com/joehalliwell/paraphernalia/actions/workflows/docs.yml/badge.svg)](https://github.com/joehalliwell/paraphernalia/actions)

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
