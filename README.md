# Medical Image Registration Benchmark

A Python benchmarking framework for comparing classical and learning-based medical image registration methods on 3D medical datasets.

This project brings together affine and deformable registration approaches, evaluation metrics, and dataset tooling in a single reproducible pipeline. It is designed for research and experimentation in medical imaging, with a strong focus on clinically relevant image alignment tasks such as CT/CT and CT/MR registration.

## Why this project matters

Medical image registration is a core problem in computer vision and healthcare AI: the goal is to align moving images to fixed images so that anatomy can be compared, analyzed, or used downstream in segmentation, treatment planning, and motion correction.

This repository is useful for:

- benchmarking multiple registration methods under the same conditions
- comparing classical algorithms with deep learning approaches
- evaluating image alignment quality with medical metrics
- experimenting with deformable fields and displacement analysis
- building a reusable pipeline for imaging research or production prototypes

## Included methods

The benchmark includes multiple registration strategies, including:

- ANTs SyN
- NiftyReg affine and B-spline registration
- SimpleITK Demons
- corrField
- VoxelMorph
- LapIRN
- HyperMorph

## Repository structure

- `registrationbaselines/registration/` — method implementations and registration interfaces
- `registrationbaselines/data_loading/` — dataset loaders and pre-processing hooks
- `registrationbaselines/evaluation/` — metrics, plotting, CSV output, and result analysis
- `registrationbaselines/displacement/` — displacement field and deformation utilities
- `registrationbaselines/configs/` — YAML configs for registration methods
- `examples/` — runnable end-to-end scripts
- `registrationbaselines/tests/` — validation and regression checks
- `docs/` — project documentation notes

## Skills demonstrated

This work sits at the intersection of:

- Python and scientific computing
- medical imaging and 3D data processing
- classical optimization and image registration algorithms
- deep learning for biomedical image analysis
- experiment tracking and reproducible benchmarking
- evaluation design using quantitative imaging metrics

## Public 2D cardiac MRI example

This repository includes a lightweight end-to-end registration demo built around a 2D cardiac MRI example using the provided NIfTI pair or the public ACDC cardiac challenge dataset.

By default, the demo searches the project `tmp/` folder for `acdc_fixed.nii.gz` and `acdc_moving.nii.gz` and registers that provided pair directly. If the pair is absent, it can use a local ACDC dataset via `ACDC_DATASET_PATH`, or fall back to a synthetic cardiac phantom. The demo reports alignment metrics and saves fixed, moving, and registered PNGs for inspection.

Generated demo artifacts are saved under `tmp/public_2d_demo_outputs/`, which is intentionally kept out of git so local benchmark output remains easy to inspect without polluting the repository.

![Fixed cardiac slice](tmp/public_2d_demo_outputs/fixed.png)

![Moving image before registration](tmp/public_2d_demo_outputs/moving_before_registration.png)

![Moving image after registration](tmp/public_2d_demo_outputs/moving_after_registration.png)

```bash
pip install -r requirements.txt
python examples/public_2d_acdc_demo.py --output-dir tmp/public_2d_demo_outputs
```

If you have the ACDC dataset locally, you can point the script at it with:

```bash
ACDC_DATASET_PATH=/path/to/acdc python examples/public_2d_acdc_demo.py --output-dir tmp/public_2d_demo_outputs
```

The script saves fixed, moving, and aligned image outputs under the chosen output directory for inspection and comparison.

## Architecture and design overview

```mermaid
flowchart LR
    A[Public medical image dataset] --> B[Data loading]
    B --> C[Registration pipeline]
    C --> D[Rigid transform estimation]
    D --> E[Warped image]
    E --> F[Similarity metric + diagnostics]
    F --> G[Results / PNG outputs]
    H[Config files] --> C
```

The project is built around a modular pipeline: dataset loaders feed image pairs into a shared registration interface, the deformation or displacement is computed, and the results are evaluated using consistent metrics and saved artifacts. This design makes the code easier to explain in interviews, easier to extend in research, and clearer as a real engineering workflow.

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run an example pipeline

```bash
python examples/fullPipeline.py
```

This example loads a dataset, runs a registration method, and evaluates the alignment quality.

## Example usage

```python
from pathlib import Path
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.registration.demons_sitk import DemonsSITK

base_dir = Path(".")
config_path = base_dir / "registrationbaselines/configs/DemonsSITK.yaml"

loader = data_loaders.L2RLungCTDataset(
    Path("/path/to/dataset"),
    return_type="path_dict",
    indices=[0, 1],
)

registration = DemonsSITK(config_path, loader, use_masked_evaluation=True)
registration.evaluate_with_zero_displacement()
registration.execute_with_one_parameter_set()
```

## Data expectations

This project is built for medical imaging datasets in folder structures compatible with the included dataloaders. In the examples, datasets are passed as paths such as:

- Lung CT benchmarks
- abdominal CT/CT pairs
- MR/CT registration tasks

A working installation generally requires preprocessed NIfTI images and a dataset directory consistent with the loader definitions.

## Evaluation and benchmarking

The framework supports:

- quantitative comparison across registration algorithms
- displacement field inspection
- result plotting and CSV export
- mask-aware evaluation workflows
- zero-displacement baselines for sanity checks

## Project status

This repository is a research-oriented benchmarking project rather than a polished SaaS product. It is well suited for demonstrating:

- research engineering skills
- strong Python implementation ability
- familiarity with medical imaging pipelines
- experimentation and benchmarking mindset

## License

This project is distributed under the license found in the repository: `LICENSE`.



