# Loudness Control and Audio Transfer Function Analysis

Welcome to the **Loudness** app — an interactive Streamlit application designed to analyze and manipulate audio signals based on frequency-specific amplitude control.

## Overview

This app allows users to:

- **Adjust signal amplitudes** at multiple frequencies using interactive sliders.
- **Generate and listen to sinusoidal signals** corresponding to those amplitudes.
- **Visualize amplitude vs. frequency curves** in decibels (dB).
- **Calculate the transfer function** between two sets of amplitudes and display amplitude differences.
- **Estimate the impulse response** by reconstructing the symmetric spectrum and applying inverse FFT.
- **Convolve audio files** with the estimated impulse responses for audio effect simulation.
- **Listen to the original and convolved audio results** directly within the app.

## Features

- Interactive frequency amplitude sliders for two columns (sets) of frequencies.
- Real-time audio generation and playback of sine wave signals.
- Graphical visualization of amplitude-frequency relationships and transfer functions.
- Computation of transfer functions (difference between amplitude sets) in dB.
- Symmetric spectrum reconstruction and impulse response estimation via IFFT.
- Audio convolution using FFT-based methods on stereo audio input.
- Visualization of convolved audio signals in the time domain (amplitude in dB).
- Playback of original and processed audio signals.

## Technologies Used

- **Python**  
- **Streamlit** — for creating the interactive web app.  
- **NumPy** and **SciPy** — for signal processing and numerical computations.  
- **Matplotlib** — for plotting graphs and visualizations.  
- **SoundFile** — for reading/writing audio files.

## How to Use

1. Access the app at: [https://loudness.streamlit.app/](https://loudness.streamlit.app/)

2. Use the sliders in the two columns to set amplitudes for different frequencies.

3. Listen to the generated sine wave signals for each frequency/amplitude.

4. View the amplitude vs. frequency curves and the calculated transfer function.

5. Upload or select an audio file to apply convolution with the estimated impulse responses.

6. Listen to the original and convolved audio outputs to hear the effect of amplitude adjustments.
