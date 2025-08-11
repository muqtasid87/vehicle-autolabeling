# Vehicle Inference Package

A Python package for vehicle detection and classification using YOLO + VLM (Qwen or Gemma).

## Installation
pip install .

## Usage
vehicle-inference --model qwen --input-folder /path/to/images --output-folder /path/to/json --use-finetuned --cot --batch-size 4

For help: vehicle-inference --help

## Features
- Supports Qwen or Gemma VLMs.
- OOP design.
- Batched inference.
- Auto-downloads YOLO model if missing.