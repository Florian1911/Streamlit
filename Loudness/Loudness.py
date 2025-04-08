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

# Lecture du fichier audio
st.audio(audio_path, format="audio/wav")
sample_rate, data = wavfile.read(audio_path)


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

# Génération du signal Chirp exponentiel
st.header("Chirp Exponentiel et Analyse")

# Entrées de l'utilisateur pour personnaliser le signal Chirp
freq_start = st.slider("Fréquence de départ (Hz)", 10, 1000, 30,help="Sélectionnez la fréquence de départ du chirp exponentiel")
freq_end = st.slider("Fréquence de fin (Hz)", 1000, 20000, 10000,help="Sélectionnez la fréquence de fin du chirp exponentiel")
dur = st.slider("Durée (s)", 1, 20, 10,help="Sélectionnez la durée du chirp exponentiel")

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
st.header("Loudness Test - Amplitude Perception",help="Sélectionnez l'amplitude du signal pour chaque fréquence afin que chaque signal soit perçu de la même amplitude")

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
        st.write(f"Fréquence : {freq} Hz")

        # Slider pour contrôler l'amplitude réelle du signal
        amplitude_selectionnee = st.slider(
            f"Amplitude pour {freq} Hz", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"col1_{i}",help="Sélectionnez l'amplitude du signal pour cette fréquence"
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
        st.write("Courbe de l'amplitude en fonction de la fréquence (Colonne 1)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col1_db, marker='o', linestyle='-', color='b')
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Courbe de l'Amplitude en fonction de la Fréquence (Colonne 1)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("Veuillez sélectionner une amplitude pour chaque fréquence dans la colonne 1.")

# Deuxième colonne
with col2:
    for i, freq in enumerate(FREQUENCIES):
        st.write(f"Fréquence : {freq} Hz")

        # Slider pour contrôler l'amplitude réelle du signal
        amplitude_selectionnee = st.slider(
            f"Amplitude pour {freq} Hz", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"col2_{i}",help="Sélectionnez l'amplitude du signal pour cette fréquence"
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
        st.write("Courbe de l'amplitude en fonction de la fréquence (Colonne 2)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col2_db, marker='o', linestyle='-', color='r')
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Courbe de l'Amplitude en fonction de la Fréquence (Colonne 2)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("Veuillez sélectionner une amplitude pour chaque fréquence dans la colonne 2.")


# Calculer la fonction de transfert entre les deux colonnes
if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
    transfer_function_db = amplitudes_selectionnees_col2_db - amplitudes_selectionnees_col1_db
    st.header("Fonction de transfert entre les deux colonnes")
    plt.figure(figsize=(8, 6))
    plt.plot(FREQUENCIES, transfer_function_db, marker='o', linestyle='-', color='g')
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Différence d'Amplitude (dB)")
    plt.title("Fonction de transfert entre les deux courbes")
    plt.grid(True)
    st.pyplot(plt)

    st.header("Inverse FFT de la Fonction de Transfert")

    # 1. Convertir la fonction de transfert de dB à l'échelle linéaire
    transfer_function_linear = 10**(transfer_function_db / 20)

    # 2. Créer un spectre symétrique
    # On prend la moitié du spectre (fréquences positives)
    positive_spectrum = transfer_function_linear

    negative_spectrum = positive_spectrum[::-1] # On inverse l'ordre
    positive_spectrum = np.insert(positive_spectrum, 0, 1)

    negative_frequency = -np.array(FREQUENCIES[::-1])
    positive_frequency = np.array(FREQUENCIES)
    positive_frequency = np.insert(positive_frequency, 0, 0)
    DOUBLE_FREQUENCIES = np.concatenate((negative_frequency,positive_frequency))

    full_spectrum_linear = np.concatenate((negative_spectrum,positive_spectrum))

    # Afficher le spectre symétrique
    st.subheader("Spectre Symétrique Linéaire")
    plt.figure(figsize=(8, 6))
    plt.plot(DOUBLE_FREQUENCIES,full_spectrum_linear)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectre Symétrique")
    plt.grid(True)
    st.pyplot(plt)

    # 4. Calculer l'IFFT
    time_domain_response = np.fft.ifft(full_spectrum_linear)
    time_domain_response_db = 20 * np.log10(np.abs(time_domain_response))
    # Afficher la réponse impulsionnelle
    st.subheader("Réponse Impulsionnelle Estimée")
    plt.figure(figsize=(8, 6))
    plt.plot((time_domain_response_db))
    plt.xlabel("Temps")
    plt.ylabel("Amplitude (dB)")
    plt.title("Réponse Impulsionnelle (IFFT de la Fonction de Transfert)")
    plt.grid(True)
    st.pyplot(plt)

st.subheader("🔊 Son original")
st.audio("beth.wav", format="audio/wav")
sample_rate, data = wavfile.read("beth.wav")
convolved = fftconvolve(data.astype(np.float32), time_domain_response, mode="full")
convolved = convolved / np.max(np.abs(convolved))
convolved = (convolved * 32767).astype(np.int16)

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
    wavfile.write(tmpfile.name, sample_rate, convolved)
    st.subheader("🔊 Résultat convolué")
    st.audio(tmpfile.name, format="audio/wav")



with st.expander("Explications"):
    st.markdown('''La convolution numérique est définie comme''')
    st.latex("y(n)=x(n)*h(n)=\sum_{k=-\infty}^{\infty}x(k)h(n-k)")
