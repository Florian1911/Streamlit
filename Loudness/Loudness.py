import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy as sc
import librosa
import soundfile as sf
import io
import libfmp.b
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import fftconvolve
import tempfile
import os

# Chemin absolu depuis le fichier Python courant
base_dir = os.path.dirname(__file__)
audio_path = os.path.join(base_dir, "assets", "beth.wav")

# Fonction pour générer le signal Chirp exponentiel
def generate_chirp_exp(dur, freq_start, freq_end, Fs=44100):
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    freq = np.exp(np.linspace(np.log(freq_start), np.log(freq_end), N))
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n-1] + 2 * np.pi * freq[n-1] / Fs
    x = np.sin(phases)
    return x, t, freq

def read_audio_from_wav(audio_file):
    audio_file.seek(0)
    signal, sample_rate = sf.read(audio_file)
    return signal, sample_rate

# Enregistrement du signal audio dans un fichier temporaire .wav
def save_audio_to_wav(signal, sample_rate, filename="output.wav"):
    """Sauvegarde le signal en fichier WAV sans le normaliser"""
    signal_int16 = np.int16(signal)
    write(filename, sample_rate, signal_int16)
    return filename

# Fréquence d'échantillonnage
Fs = 44100

st.title("Hear with my ears")

st.write("Rajouter une introduction")

# Génération du signal Chirp exponentiel
st.header("Exponential Chirp and Analysis")

# Entrées de l'utilisateur pour personnaliser le signal Chirp

freq_start = 30
freq_end=18000
dur=10

x, t, freq = generate_chirp_exp(dur, freq_start=freq_start, freq_end=freq_end, Fs=Fs)

fig, ax = plt.subplots(1, 1, gridspec_kw={'width_ratios': [2]}, figsize=(7, 3))

# Transformée de Fourier pour afficher le spectrogramme
N, H = 1024, 512
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N)
libfmp.b.plot_matrix(np.log(1 + np.abs(X)), Fs=Fs / H, Fs_F=N / Fs, ax=[ax],
                     title='Spectrogram of chirp', colorbar=False)

plt.tight_layout()
st.pyplot(fig)

# Lecture du son
st.audio(x, sample_rate=Fs)

# Test d'amplitude
st.header("Loudness Test - Amplitude Perception", help="Select the signal amplitude for each frequency so that each signal is perceived with the same amplitude")

# Paramètres de fréquence et durée
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000,6000, 8000,10000,12000,14000, 16000]
Fs = 44100  # Fréquence d'échantillonnage
DURATION = 2  # Durée du son en secondes

amplitudes_selectionnees_col1 = []
amplitudes_selectionnees_col2 = []

# Liste des amplitudes possibles
amplitudes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]

# Liste pour stocker les amplitudes sélectionnées
amplitudes_selectionnees = []

MAX_INT16 = 32767
DURATION = 2

col1, col2 = st.columns(2)

with col1:
    for i, freq in enumerate(FREQUENCIES):
        st.write(f"Frequency : {freq} Hz")

        # Slider pour contrôler l'amplitude réelle du signal
        amplitude_selectionnee = st.slider(
            f"Amplitude for {freq} Hz", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"col1_{i}",help="Select the signal amplitude for this frequency"
        )
        amplitudes_selectionnees_col1.append(amplitude_selectionnee)

        # Génération du signal avec l'amplitude choisie
        temps = np.linspace(0, DURATION, int(Fs * DURATION), endpoint=False)
        signal = amplitude_selectionnee * MAX_INT16 * np.sin(2 * np.pi * freq * temps)

        # Sauvegarde du fichier audio avec l'amplitude réelle
        audio_file = save_audio_to_wav(signal, Fs)

        # Affichage de l'audio avec amplitude ajustable
        st.audio(audio_file, format="audio/wav")
        # Affichage de la courbe Amplitude vs Fréquence pour la première colonne
    amplitudes_selectionnees_col1_db = 20 * np.log10(np.array(amplitudes_selectionnees_col1))

    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES):
        st.write("Amplitude versus frequency curve (Column 1)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col1_db, marker='o', linestyle='-', color='b')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Amplitude versus frequency curve (Column 1)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("Please select an amplitude for each frequency in column 1.")

# Deuxième colonne
with col2:
    for i, freq in enumerate(FREQUENCIES):
        st.write(f"Frequency : {freq} Hz")

        # Slider pour contrôler l'amplitude réelle du signal
        amplitude_selectionnee = st.slider(
            f"Amplitude for {freq} Hz", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"col2_{i}",help="Select the signal amplitude for this frequency"
        )
        amplitudes_selectionnees_col2.append(amplitude_selectionnee)

        # Génération du signal avec l'amplitude choisie
        temps = np.linspace(0, DURATION, int(Fs * DURATION), endpoint=False)
        signal = amplitude_selectionnee * MAX_INT16 * np.sin(2 * np.pi * freq * temps)

        # Sauvegarde du fichier audio avec l'amplitude réelle
        audio_file = save_audio_to_wav(signal, Fs)

        # Affichage de l'audio avec amplitude ajustable
        st.audio(audio_file, format="audio/wav")
        # Affichage de la courbe Amplitude vs Fréquence pour la deuxième colonne
    amplitudes_selectionnees_col2_db = 20 * np.log10(np.array(amplitudes_selectionnees_col2))    
    if len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        st.write("Amplitude versus frequency curve (Column 2)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col2_db, marker='o', linestyle='-', color='r')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Amplitude versus frequency curve (Column 2)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("Please select an amplitude for each frequency in column 2.")


# Calculer la fonction de transfert entre les deux colonnes
with col1:
    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        transfer_function_db1 = amplitudes_selectionnees_col2_db - amplitudes_selectionnees_col1_db
        st.header("Amplitude difference to be applied (2-1)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, transfer_function_db1, marker='o', linestyle='-', color='g')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude Difference (dB)")
        plt.title("Amplitude difference to be applied (2-1)")
        plt.grid(True)
        st.pyplot(plt)

        st.header("Inverse FFT")

        # 1. Convertir la fonction de transfert de dB à l'échelle linéaire
        transfer_function_linear1 = 10**(transfer_function_db1 / 20)

        # 2. Créer un spectre symétrique
        # On prend la moitié du spectre (fréquences positives)
        positive_spectrum1 = transfer_function_linear1

        negative_spectrum1 = positive_spectrum1[::-1] # On inverse l'ordre
        positive_spectrum1 = np.insert(positive_spectrum1, 0, 1)

        negative_frequency1 = -np.array(FREQUENCIES[::-1])
        positive_frequency1= np.array(FREQUENCIES)
        positive_frequency1 = np.insert(positive_frequency1, 0, 0)
        DOUBLE_FREQUENCIES = np.concatenate((negative_frequency1,positive_frequency1))

        full_spectrum_linear1 = np.concatenate((negative_spectrum1,positive_spectrum1))

        # Afficher le spectre symétrique
        st.subheader("Linear Symmetric Spectrum")
        plt.figure(figsize=(8, 6))
        plt.plot(DOUBLE_FREQUENCIES,full_spectrum_linear1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Linear Spectrum")
        plt.grid(True)
        st.pyplot(plt)

        # 4. Calculer l'IFFT
        time_domain_response1 = np.fft.ifft(full_spectrum_linear1)
        time_domain_response_db1 = 20 * np.log10(np.abs(time_domain_response1))
        # Afficher la réponse impulsionnelle
        st.subheader("Estimated Impulse Response")
        plt.figure(figsize=(8, 6))
        plt.plot((time_domain_response_db1))
        plt.xlabel("Time")
        plt.ylabel("Amplitude (dB)")
        plt.title("Impulse Response")
        plt.grid(True)
        st.pyplot(plt)


with col2:
    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        transfer_function_db2 = amplitudes_selectionnees_col1_db - amplitudes_selectionnees_col2_db
        st.header("Amplitude difference to be applied (1-2)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, transfer_function_db2, marker='o', linestyle='-', color='g')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude Difference (dB)")
        plt.title("Amplitude difference to be applied (1-2)")
        plt.grid(True)
        st.pyplot(plt)

        st.header("Inverse FFT")

        # 1. Convertir la fonction de transfert de dB à l'échelle linéaire
        transfer_function_linear2 = 10**(transfer_function_db2 / 20)

        # 2. Créer un spectre symétrique
        # On prend la moitié du spectre (fréquences positives)
        positive_spectrum2 = transfer_function_linear2

        negative_spectrum2 = positive_spectrum2[::-1] # On inverse l'ordre
        positive_spectrum2 = np.insert(positive_spectrum2, 0, 1)

        negative_frequency2 = -np.array(FREQUENCIES[::-1])
        positive_frequency2= np.array(FREQUENCIES)
        positive_frequency2 = np.insert(positive_frequency2, 0, 0)
        DOUBLE_FREQUENCIES = np.concatenate((negative_frequency2,positive_frequency2))

        full_spectrum_linear2 = np.concatenate((negative_spectrum2,positive_spectrum2))

        # Afficher le spectre symétrique
        st.subheader("Linear Symmetric Spectrum")
        plt.figure(figsize=(8, 6))
        plt.plot(DOUBLE_FREQUENCIES,full_spectrum_linear2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Symetric Spectrum")
        plt.grid(True)
        st.pyplot(plt)

        # 4. Calculer l'IFFT
        time_domain_response2 = np.fft.ifft(full_spectrum_linear2)
        time_domain_response_db2 = 20 * np.log10(np.abs(time_domain_response2))
        # Afficher la réponse impulsionnelle
        st.subheader("Estimated Impulse Response")
        plt.figure(figsize=(8, 6))
        plt.plot((time_domain_response_db2))
        plt.xlabel("Time")
        plt.ylabel("Amplitude (dB)")
        plt.title("Impulse Response")
        plt.grid(True)
        st.pyplot(plt)



st.subheader("🔊 Original sound")
st.audio(audio_path, format="audio/wav")
sample_rate, data = wavfile.read(audio_path)

col3,col4=st.columns(2)

with col3:
    st.subheader("🔊 Convolt result")
    st.write("Son a tester")
    st.write("Convolution du son original avec la valeur en dB de la première colonne (le premier graphique)")
    convolved11=fftconvolve(data,amplitudes_selectionnees_col1_db,mode="full")

    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved11)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    st.audio(convolved11,sample_rate=Fs)

    #On veut equal loudness ou equal db ?
    convolved1 = fftconvolve(data, time_domain_response1, mode="full")
    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 2")
    st.audio(convolved1,sample_rate=Fs)

with col4:
    st.subheader("🔊 Convolt result")
    convolved22 = fftconvolve(data,amplitudes_selectionnees_col2_db) 

    st.write("Son a tester")
    

    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved22)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 2")
    plt.grid(True)
    st.pyplot(plt)

    

    st.audio(convolved22,sample_rate=Fs)
    convolved2 = fftconvolve(data, time_domain_response2, mode="full")
    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 1")
    st.audio(convolved2,sample_rate=Fs)


with st.expander("Explications"):
    st.markdown('''La convolution numérique est définie comme''')
    st.latex("y(n)=x(n)*h(n)=\sum_{k=-\infty}^{\infty}x(k)h(n-k)")

