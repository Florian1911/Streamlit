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

def save_audio_to_wav2(signal, sample_rate):
    audio_file = io.BytesIO()
    sf.write(audio_file, signal, sample_rate, format='WAV')
    audio_file.seek(0)
    return audio_file

# Fréquence d'échantillonnage
Fs = 44100

# Génération du signal Chirp exponentiel

st.title("Chirp Exponentiel et Analyse")

# Entrées de l'utilisateur pour personnaliser le signal Chirp
freq_start = st.slider("Fréquence de départ (Hz)", 10, 1000, 30)
freq_end = st.slider("Fréquence de fin (Hz)", 1000, 20000, 10000)
dur = st.slider("Durée (s)", 1, 20, 10)

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
st.audio(save_audio_to_wav2(x, Fs), format="audio/wav")

# Test d'amplitude
st.title("Loudness Test - Amplitude Perception")

# Paramètres de fréquence et durée
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000,10000,12000,14000, 16000]
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
            f"Amplitude pour {freq} Hz", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"col1_{i}"
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
            f"Amplitude pour {freq} Hz", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"col2_{i}"
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
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col2, marker='o', linestyle='-', color='r')
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Courbe de l'Amplitude en fonction de la Fréquence (Colonne 2)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("Veuillez sélectionner une amplitude pour chaque fréquence dans la colonne 2.")


# Calculer la fonction de transfert entre les deux colonnes
if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
    transfer_function_db = amplitudes_selectionnees_col2_db - amplitudes_selectionnees_col1_db
    st.write("Fonction de transfert entre les deux colonnes")
    plt.figure(figsize=(8, 6))
    plt.plot(FREQUENCIES, transfer_function_db, marker='o', linestyle='-', color='g')
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Différence d'Amplitude (dB)")
    plt.title("Fonction de transfert entre les deux courbes")
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write("Veuillez sélectionner une amplitude pour chaque fréquence dans les deux colonnes.")