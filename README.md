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
- Oblique strategy of the day during startup

## review

- Move reviewed folders to a target folder "lost"
- "kept" and "lost" should excluded
- Notebook mode?

### glsl

- Support all/more Book of Shaders uniforms
- Support all Shadertoy uniforms (see https://github.com/iY0Yi/ShaderBoy)
- Support buffers

#### clip

- Is anti-prompt logic actually working?
- Adaptive focus
- Factor our perceptual modes
- Perceptual masking for CLIP
- Image prompts
- Add SRCNN
- Use https://github.com/assafshocher/ResizeRight

### torch

- Fix replace_grad and add tests
- Fix clamp_with_grad and add tests
- Add BigGAN generators
- Add soft_permutation()
- Add ZX Spectrum/C64 Standard Mode generator
- Main entry point for generator+CLIP?
- Add standard description string/slug to generators
- Add Diffusion generator
