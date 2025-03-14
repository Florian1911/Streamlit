import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy as sc
import librosa
import soundfile as sf
import io
import libfmp.b

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

# Enregistrement du signal audio dans un fichier temporaire .wav
def save_audio_to_wav(signal, sample_rate=Fs):
    # Créer un fichier temporaire en mémoire pour l'audio
    audio_file = io.BytesIO()
    sf.write(audio_file, signal, sample_rate, format='WAV')
    audio_file.seek(0)
    return audio_file

# Lecture du son
st.audio(save_audio_to_wav(x, Fs), format="audio/wav")

# Test d'amplitude
st.title("Loudness Test - Amplitude Perception")

# Paramètres de fréquence et durée
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000,10000,12000,14000, 16000]
Fs = 44100  # Fréquence d'échantillonnage
DURATION = 2  # Durée du son en secondes

# Liste des amplitudes possibles
amplitudes = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]

# Liste pour stocker les amplitudes sélectionnées
amplitudes_selectionnees = []

# Pour chaque fréquence, permettre la sélection d'une amplitude
#for freq in FREQUENCIES:
   # st.write(f"Fréquence: {freq} Hz")
    #amplitude_selectionnee = st.selectbox(f"Amplitude pour {freq} Hz", amplitudes, index=3)  # Choix par défaut : 0.5
    #amplitude_selectionnee = st.slider("Amplitude pour {freq}Hz",0,10,1)
    #if amplitude_selectionnee is not None:
        #amplitudes_selectionnees.append(amplitude_selectionnee)

        # Générer le signal avec l'amplitude choisie
        #temps = np.linspace(0, DURATION, int(Fs * DURATION), endpoint=False)
        #signal = amplitude_selectionnee * np.sin(2 * np.pi * freq * temps)
        
        # Lecture du fichier audio dans Streamlit
        #st.audio(signal,sample_rate=Fs)

DURATION = 2
for freq in FREQUENCIES:
    st.write(f"Fréquence : {freq} Hz")
    amplitude_selectionnee = st.slider(f"Amplitude pour {freq}Hz",0.0,10.0,1.0,step=0.1)

    #echantillons = np.arange(0, DURATION, 1/Fs)
    temps = np.linspace(0,2,10*Fs)
    signal = amplitude_selectionnee * np.sin(2 * np.pi * freq * temps)

    fig, ax = plt.subplots()
    ax.plot(temps, signal,'xr')
    ax.plot(temps, signal)
    ax.set_xlabel('Temps (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Signal pour {freq} Hz')
    ax.set_xlim(0, 0.001)
    ax.set_ylim(-1.1, 1.1)
    st.pyplot(fig)



    audio_file = save_audio_to_wav(signal, Fs)
    st.audio(audio_file, format="audio/wav")  
    st.audio(signal,sample_rate=Fs)

    # Calculer la FFT du signal lu depuis le fichier
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_signal), 1/Fs)

    # Tracer la FFT
    fig, ax = plt.subplots()
    ax.plot(freqs[:len(freqs)//2], np.abs(fft_signal)[:len(fft_signal)//2])
    ax.set_xlabel('Fréquence (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Spectre de fréquence pour {freq} Hz')
    st.pyplot(fig)





# Affichage de la courbe Amplitude vs Fréquence
if len(amplitudes_selectionnees) == len(FREQUENCIES):
    st.write("Courbe de l'amplitude en fonction de la fréquence")
    plt.figure(figsize=(8, 6))
    plt.plot(FREQUENCIES, amplitudes_selectionnees, marker='o', linestyle='-', color='b')
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Courbe de l'Amplitude en fonction de la Fréquence")
    plt.grid(True)
    st.pyplot(plt)
else:
    st.write("Veuillez sélectionner une amplitude pour chaque fréquence.")
