# Glossary of Abbreviations

This glossary defines all abbreviations used in the Menial AI project, particularly in the training notebook and related code.

---

## Symbols & Numbers

- **2D** - Two-Dimensional — Having width and height but no depth, like a flat image or grid.

## A

- **A100** - NVIDIA A100 (GPU model) — A high-performance data center GPU designed for AI training and inference workloads.
- **Adam** - Adaptive Moment Estimation (optimizer algorithm) — An optimization algorithm that adjusts learning rates individually for each parameter by tracking both the average and variance of past gradients, leading to faster and more stable training.
- **AdamW** - Adam with Weight Decay — A variant of Adam that applies weight decay (a penalty on large weights) separately from the gradient update, improving regularization and generalization.
- **AMP** - Automatic Mixed Precision — A training technique that uses a mix of 16-bit and 32-bit floating-point numbers to speed up computation and reduce memory usage while maintaining model accuracy.
- **API** - Application Programming Interface — A set of rules and protocols that allows different software programs to communicate with each other.
- **AST** - Audio Spectrogram Transformer — A model architecture that applies the transformer (attention-based) approach to audio spectrograms for sound classification tasks.
- **AudioSet** - Google's large-scale audio dataset (2M+ YouTube clips, 521 classes) — A massive collection of labeled audio clips sourced from YouTube, commonly used as a benchmark for training and evaluating audio classification models.

## B

- **BatchNorm** - Batch Normalization (neural network layer that normalizes activations) — A technique that normalizes the inputs to each layer within a mini-batch during training, stabilizing and accelerating the learning process.

## C

- **C++** - C plus plus (programming language) — A general-purpose, high-performance programming language widely used in systems software, game engines, and performance-critical applications.
- **CD** - Compact Disc — An optical disc format originally designed for storing digital audio, with a standard sample rate of 44.1 kHz.
- **CCA** - Connected Component Analysis — A technique that identifies and labels distinct connected regions in a binary or labeled image (or matrix), grouping adjacent elements that share the same value into individually addressable components.
- **CNN** - Convolutional Neural Network — A type of neural network that uses small sliding filters to automatically detect spatial patterns (like edges, textures, and shapes) in grid-structured data such as images or spectrograms.
- **Conv2d** - 2D Convolution (convolutional layer for 2D data) — A neural network layer that slides a small filter across a two-dimensional input (such as an image or spectrogram) to extract local features.
- **CPU** - Central Processing Unit — The primary general-purpose processor in a computer that executes instructions and handles most computational tasks.
- **CSV** - Comma-Separated Values — A simple, plain-text file format where each line represents a row of data and values within each row are separated by commas.
- **CUDA** - Compute Unified Device Architecture (NVIDIA's parallel computing platform) — A software platform by NVIDIA that lets developers run general-purpose computations on NVIDIA GPUs, enabling massive parallelism for tasks like deep learning.
- **cuDNN** - CUDA Deep Neural Network library — An NVIDIA library of optimized building blocks (such as convolutions and pooling) that accelerates deep learning operations on NVIDIA GPUs.

## D

- **dB** - Decibels (unit of sound intensity) — A logarithmic unit used to measure the intensity or loudness of a sound, where each 10 dB increase represents roughly a doubling in perceived loudness.

## E

- **e.g.** - exempli gratia (Latin: "for example") — Used to introduce one or more illustrative examples.
- **ESC-50** - Environmental Sound Classification - 50 categories (audio dataset) — A benchmark dataset containing 2,000 short environmental audio recordings organized into 50 classes such as rain, dog bark, and clock tick.
- **etc.** - et cetera (Latin: "and so forth") — Used to indicate that a list continues with similar items.

## F

- **F1** - F1 Score (harmonic mean of precision and recall) — A single metric that balances precision (how many predicted positives are correct) and recall (how many actual positives are found), where 1.0 is perfect and 0.0 is worst.
- **FFT** - Fast Fourier Transform — An efficient algorithm that converts a signal from the time domain into its constituent frequencies, revealing which frequencies are present and how strong they are.

## G

- **GPU** - Graphics Processing Unit — A processor with thousands of small cores designed for parallel computation, making it especially effective for training deep learning models.

## H

- **Hz** - Hertz (unit of frequency) — A unit measuring the number of cycles per second; for example, a 440 Hz tone means the sound wave oscillates 440 times per second.

## I

- **i.e.** - id est (Latin: "that is") — Used to clarify or restate something in different words.
- **ID** - Identifier — A unique label or code used to distinguish one item (such as a data sample, class, or object) from another.
- **inf** - infinity (mathematical concept; infinite value) — A special value representing a number larger than any finite number, often encountered in computations involving division by zero or unbounded growth.
- **ISTFT** - Inverse Short-Time Fourier Transform — The reverse of the STFT; it reconstructs a time-domain audio signal from its frequency-over-time representation (spectrogram).

## J

- **JSON** - JavaScript Object Notation — A lightweight, human-readable text format for storing and exchanging structured data using key–value pairs and arrays.

## K

- **KB** - Kilobytes — A unit of digital data size equal to 1,024 bytes (or 1,000 bytes in decimal convention).

## L

- **L2** - L2 norm/regularization (method for preventing overfitting) — A regularization technique that adds a penalty proportional to the squared magnitude of model weights, discouraging excessively large values and helping the model generalize to new data.
- **LLM** - Large Language Model — A neural network trained on vast amounts of text data that can understand and generate human-like language, such as GPT or Claude.
- **LR** - Learning Rate — A hyperparameter that controls how much a model's weights are adjusted in response to each gradient update; too high risks instability, too low risks slow training.

## M

- **matplotlib** - Python plotting library (Mathematical Plotting Library) — A widely used Python library for creating static, animated, and interactive charts, graphs, and visualizations.
- **MaxPool** - Max Pooling (downsampling operation that selects maximum values) — A downsampling layer that reduces the spatial size of its input by selecting the maximum value within each small region, retaining the most prominent features.
- **MB** - Megabytes — A unit of digital data size equal to 1,024 kilobytes (roughly one million bytes).
- **MFCC** - Mel-Frequency Cepstral Coefficients — A compact set of audio features that represent the shape of the spectral envelope on a perceptually motivated (mel) frequency scale, widely used in speech and audio classification.
- **ms** - milliseconds — One thousandth of a second (0.001 s); a common time unit for measuring short audio events and processing latencies.

## N

- **NaN** - Not a Number (indicates an undefined or unrepresentable numerical result) — A special floating-point value that signals an invalid or undefined result, such as dividing zero by zero.
- **NLL** - Negative Log-Likelihood — A loss function that measures how well a probability model's predictions match the true labels; lower values indicate better predictions.
- **NMF** - Non-Negative Matrix Factorization — A technique that decomposes a matrix into two smaller matrices with all non-negative values, useful for discovering additive parts or patterns in data such as audio spectra.
- **numpy** - Numerical Python (scientific computing library) — The foundational Python library for fast, efficient operations on multi-dimensional arrays and matrices, underpinning most scientific and ML code in Python.

## O

- **ONNX** - Open Neural Network Exchange — An open standard format for representing trained machine learning models, enabling models to be transferred between different frameworks (e.g., PyTorch to TensorFlow).

## P

- **pandas** - Python Data Analysis Library (data manipulation library) — A Python library that provides flexible data structures (DataFrames and Series) for efficiently loading, cleaning, transforming, and analyzing tabular data.
- **PANNs** - Pretrained Audio Neural Networks — A family of CNN-based models pretrained on AudioSet for general-purpose audio tagging and classification, commonly used as feature extractors or fine-tuning starting points.
- **pt** - PyTorch (file extension) — The standard file extension for saved PyTorch model weights and tensors.
- **PyTorch** - Python-based deep learning framework — An open-source machine learning library developed by Meta that provides flexible tensor computation and automatic differentiation, widely used for research and production deep learning.

## R

- **ReLU** - Rectified Linear Unit (activation function) — A simple activation function that outputs the input directly if it is positive and zero otherwise (i.e., max(0, x)), helping neural networks learn non-linear patterns.
- **ResNet** - Residual Network (deep neural network architecture with skip connections) — A deep CNN architecture that uses shortcut (skip) connections to let gradients flow directly through layers, enabling the effective training of very deep networks (50, 101, or more layers).
- **RGB** - Red Green Blue (color channels) — A color model that represents images using three channels—red, green, and blue—whose intensities combine to produce a wide range of colors.
- **RMSprop** - Root Mean Square Propagation (optimization algorithm) — An adaptive optimizer that divides each parameter's gradient by a running average of recent gradient magnitudes, helping maintain steady progress across parameters with different scales.
- **RNG** - Random Number Generator — An algorithm or device that produces sequences of numbers with no discernible pattern, used to ensure randomness in data shuffling, weight initialization, and augmentation.

## S

- **scikit-learn** - Python machine learning library (also imported as **sklearn**) — A popular Python library providing simple, efficient tools for common machine learning tasks such as classification, regression, clustering, and model evaluation.
- **scipy** - Scientific Python (library for scientific computing) — A Python library that builds on NumPy to provide optimized routines for signal processing, optimization, statistics, and other scientific computing tasks.
- **SGD** - Stochastic Gradient Descent — A core optimization algorithm that updates model weights using the gradient computed on a small random subset (mini-batch) of the training data at each step, trading precision for speed.
- **sklearn** - scikit-learn (common import alias) — The standard Python import name for the scikit-learn library (e.g., `import sklearn`).
- **SpecAugment** - Spectrogram Augmentation (data augmentation technique for audio) — A data augmentation method that randomly masks blocks of time steps and/or frequency bands in a spectrogram, improving model robustness without altering the original audio.
- **STFT** - Short-Time Fourier Transform — A technique that divides an audio signal into short overlapping segments and applies the Fourier Transform to each, producing a time–frequency representation (spectrogram).

## T

- **T4** - NVIDIA Tesla T4 (GPU model) — A power-efficient data center GPU commonly available in cloud environments, suitable for both training smaller models and running inference.
- **TensorFlow** - Google's open-source machine learning framework — An open-source library developed by Google for building and deploying machine learning models, offering tools for everything from research prototyping to large-scale production.

## V

- **VM** - Virtual Machine — A software-based emulation of a physical computer that runs its own operating system and applications, commonly used in cloud computing environments.
- **vs.** - versus (Latin: "against" or "compared to") — Used to indicate a comparison or contrast between two items.

## W

- **wav** - Waveform Audio File Format — An uncompressed audio file format that stores raw waveform data, preserving full audio quality at the cost of larger file sizes.

## Y

- **YAMNet** - Yet Another Mobile Network (Google's audio classification model trained on AudioSet) — A lightweight, MobileNet-based audio classification model from Google that predicts over 500 sound event classes and is commonly used for transfer learning in audio tasks.

## Z

- **ZCR** - Zero-Crossing Rate — An audio feature that counts how often a signal's waveform crosses the zero-amplitude axis per unit of time, useful for distinguishing between voiced and unvoiced sounds or percussive versus tonal content.

---

_This glossary covers abbreviations from audio signal processing, machine learning, computing hardware, data formats, libraries, Latin phrases, and general technical terms used throughout the project._
