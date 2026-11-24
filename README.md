# ML Magnetometry — Interactive Demo

An interactive web application for **machine learning-based classification of magnetic states from magnetic hysteresis loops**, developed using Python, PyTorch, and Streamlit.

The application provides a visual interface for exploring magnetic hysteresis data, comparing machine learning models, and predicting magnetic states from simulated or experimental measurements.

**[Launch the Interactive Demo](https://ml-magnetometry-demo.streamlit.app/)**

## Overview

Magnetic hysteresis loops contain valuable information about the magnetic behavior of a system. Machine learning offers an approach to identifying underlying magnetic states by recognizing patterns in these measurements.

ML Magnetometry explores this approach using neural network architectures trained to distinguish between three magnetic states:

- **Superparamagnetic (SPM):** Magnetic behavior characterized by thermally fluctuating magnetization.
- **Single-domain (SD):** Magnetic configurations characterized by a single magnetic domain.
- **Multi-domain (MD):** Magnetic configurations involving multiple magnetic domains.

The application provides an interactive environment for investigating these classifications and exploring model predictions.

## Features

### Magnetic State Classification

Predict magnetic states from magnetic hysteresis measurements using pretrained machine learning models.

The application provides:

- Classification into SPM, SD, and MD magnetic states.
- Prediction scores for each magnetic class.
- Ensemble predictions combining multiple independently trained models, when available.
- Visualization of the input hysteresis curves and classification results.

### Machine Learning Models

The application supports several neural network architectures and feature configurations.

| Architecture | Description |
|---|---|
| MLP | Multilayer Perceptron |
| MLP + Physics | MLP incorporating physics-based features |
| MLP + Preisach | MLP enhanced with Preisach-derived features |
| MLP + Physics + Preisach | MLP combining physics-based and Preisach features |
| RNN | Recurrent Neural Network |
| RNN + Physics | RNN incorporating physics-based features |
| RNN + Preisach | RNN enhanced with Preisach-derived features |
| RNN + Physics + Preisach | RNN combining physics-based and Preisach features |

Model availability depends on the pretrained checkpoints included in the deployment.

### Interactive Hysteresis Visualization

Explore magnetic hysteresis loops and their associated predictions through interactive visualizations.

Depending on the available data, the interface supports examining magnetic responses at different temperatures and comparing their behavior across measurements.

### Experimental Data Analysis

Users can upload magnetic measurements for exploratory classification.

Supported input formats include:

- MPMS3 measurement files (`.dat`).
- Compatible measurement datasets (`.csv`).

Uploaded measurements are processed by the available pretrained models to estimate the corresponding magnetic state.

Because the models are trained on simulated data, experimental measurements may introduce differences in noise, units, measurement protocols, and physical conditions that affect prediction reliability.

### Model Interpretability

The application includes tools for investigating model predictions and understanding which aspects of the input data influence classification.

Depending on the selected model, these may include:

- Integrated Gradients attribution.
- Analysis of contributions from different regions of the hysteresis curves.
- Preisach-based sensitivity analysis.
- Comparison of predictions across model architectures.

These methods support exploration of model behavior rather than providing definitive physical explanations.

## Application Architecture

The repository contains the source code required to run the interactive application, including its inference pipeline, model definitions, preprocessing utilities, and visualization components.

```text
ML-Magnetometry-Demo/
├── app.py
├── asset_paths.py
├── requirements.txt
├── .python-version
└── src/
    ├── config.py
    ├── dataset_ml/
    │   └── datasets_ml.py
    └── models/
        ├── demo.py
        ├── registry.py
        ├── basic_models.py
        ├── presentation_plots.py
        └── ...
```

The application is deployed using Streamlit Community Cloud.

Trained model checkpoints and supporting inference data are stored separately in a private Hugging Face repository and retrieved by the application when required.

This separation keeps the public repository focused on the application source code without distributing the complete research datasets or trained model weights.

## Technologies

The project is developed using:

- **Python** — Core programming language.
- **PyTorch** — Neural network architectures and model inference.
- **Streamlit** — Interactive web interface.
- **NumPy and Pandas** — Numerical processing and data manipulation.
- **Matplotlib** — Scientific visualization.
- **Scikit-learn** — Machine learning utilities.
- **Hugging Face Hub** — Private storage and retrieval of pretrained models and inference data.

## Project Scope

This repository focuses on the interactive demonstration of machine learning methods for magnetic state classification.

It includes the code required for inference, data processing, model interpretation, and visualization.

The complete research workflow, including micromagnetic simulations, dataset generation, model training, and experiment management, is maintained separately.

## Scientific Considerations

The application is intended for scientific research, demonstration, and exploratory analysis.

Predictions depend on the characteristics of the training dataset and the assumptions underlying the machine learning models.

In particular, applying models trained on simulated hysteresis curves to experimental measurements may introduce domain differences that influence classification accuracy.

Prediction scores should not automatically be interpreted as calibrated probabilities, and model outputs should be evaluated alongside experimental evidence and established physical understanding.

## License

No license is specified in this repository. All rights are reserved unless a license is explicitly added.
