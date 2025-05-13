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
from scipy.signal import lfilter

# Chemin absolu depuis le fichier Python courant
base_dir = os.path.dirname(__file__)
audio_path = os.path.join(base_dir, "assets", "stereo.wav")

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

st.markdown("""
### 🎯 Goal:
Compare how two people perceive the same sounds differently. This application allows one person to hear how the other person experiences sound.

### 🧪 Steps:
- An **exponential chirp signal** (a sweep through frequencies) is generated to analyze the overall frequency spectrum.
- For each frequency, both individuals adjust the amplitude so the sound feels equally loud to them.
- The generated curves show the differences in perception between the two columns.
- A transfer function is calculated and then **applied to a real audio sample** to simulate how it would be perceived by the other person.

Take your time to listen to each frequency, adjust based on your perception, and observe how it affects the final sound output!
""")

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
st.header("Loudness Test - Amplitude Perception", help="Adjust the amplitude so that each frequency sounds equally loud to you.")

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
        time_domain_response1 = np.fft.ifft(full_spectrum_linear1).real

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
        time_domain_response2 = np.fft.ifft(full_spectrum_linear2).real
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
left_channel = data[:, 0]
right_channel = data[:, 1]

col3,col4=st.columns(2)

with col3:
    st.subheader("🔊 Convolt result")
    # 1. Conversion dB → linéaire
    amplitudes_lin = 10**(amplitudes_selectionnees_col1_db / 20)

    # 2. Ajouter une phase nulle (sinon pas de partie imaginaire)
    spectre = amplitudes_lin

    # 3. Reconstruire la symétrie hermitienne (pour signal réel)
    spectre_complet = np.concatenate([spectre, np.conj(spectre[-2:0:-1])])

    # 4. IFFT → signal temporel
    signal_temps = np.fft.ifft(spectre_complet).real

    # 5. Convolution avec un autre signal



    # Appliquer la convolution FFT pour chaque canal
    convolved_left = fftconvolve(left_channel, signal_temps, mode="full")
    convolved_right = fftconvolve(right_channel, signal_temps, mode="full")

    # Recombiner les deux canaux
    convolved11 = np.stack([convolved_left, convolved_right], axis=1)



    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved11)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    convolved11 /= np.max(np.abs(convolved11) + 1e-6)

    buffer1 = io.BytesIO()
    sf.write(buffer1, convolved11, sample_rate, format='WAV')
    st.audio(buffer1.getvalue(), format='audio/wav')


    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 2")



    convolved_left = fftconvolve(left_channel, time_domain_response1, mode="full")
    convolved_right = fftconvolve(right_channel, time_domain_response1, mode="full")

    convolved1 = np.stack([convolved_left, convolved_right], axis=1)

    convolved1 /= np.max(np.abs(convolved1) + 1e-6)


    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved1)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)


    buffer2 = io.BytesIO()
    sf.write(buffer2, convolved1, sample_rate, format='WAV')
    st.audio(buffer2.getvalue(), format='audio/wav')



with col4:
    st.subheader("🔊 Convolt result")

    # 1. Conversion dB → linéaire
    amplitudes_lin = 10**(amplitudes_selectionnees_col2_db / 20)

    # 2. Ajouter une phase nulle (sinon pas de partie imaginaire)
    spectre = amplitudes_lin

    # 3. Reconstruire la symétrie hermitienne (pour signal réel)
    spectre_complet = np.concatenate([spectre, np.conj(spectre[-2:0:-1])])

    # 4. IFFT → signal temporel
    signal_temps = np.fft.ifft(spectre_complet).real

    # 5. Convolution avec un autre signal

    convolved_left = fftconvolve(left_channel, signal_temps, mode="full")
    convolved_right = fftconvolve(right_channel, signal_temps, mode="full")

    # Recombiner les deux canaux
    convolved22 = np.stack([convolved_left, convolved_right], axis=1)

    

    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved22)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 2")
    plt.grid(True)
    st.pyplot(plt)

    
    convolved22 /= np.max(np.abs(convolved22) + 1e-6)

    buffer1 = io.BytesIO()
    sf.write(buffer1, convolved22, sample_rate, format='WAV')
    st.audio(buffer1.getvalue(), format='audio/wav')


    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 1")



    convolved_left = fftconvolve(left_channel, time_domain_response2, mode="full")
    convolved_right = fftconvolve(right_channel, time_domain_response2, mode="full")

    # Recombiner les deux canaux
    convolved2 = np.stack([convolved_left, convolved_right], axis=1)



    plt.figure(figsize=(8, 6))
    plt.plot(20*np.log10(np.abs(convolved2)))
    plt.xlim(0,10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    convolved2 /= np.max(np.abs(convolved2) + 1e-6)

    buffer2 = io.BytesIO()
    sf.write(buffer2, convolved2, sample_rate, format='WAV')
    st.audio(buffer2.getvalue(), format='audio/wav')



with st.expander("📘 Theory explanations"):
    st.markdown("### 🎧 Why Fourier Transform?")
    st.markdown("""
To simulate how someone else hears, we analyze differences in sound perception **in the frequency domain** using the Discrete Fourier Transform (DFT):
""")
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}")
    st.markdown(r"""
- \( x[n] \): audio signal in time domain  
- \( X[k] \): frequency components  
- \( N \): number of samples
""")

    st.markdown("---")
    st.markdown("### 🔄 Transfer Function")
    st.markdown("""
Once both users have adjusted how loud each frequency feels to them, we compute the difference in decibels:
""")
    st.latex(r"\Delta A_{\text{dB}}(f) = 20 \cdot \log_{10}\left(\frac{A_2(f)}{A_1(f)}\right)")

    st.markdown("""
This gives us the **difference in perception** for each frequency \( f \). We convert this to a linear scale:
""")
    st.latex(r"H(f) = 10^{\Delta A_{\text{dB}}(f)/20}")
    st.markdown("""
\( H(f) \) is the **transfer function**, a filter that transforms audio from one hearing profile to the other.
""")

    st.markdown("---")
    st.markdown("### ⏪ Back to Time Domain")
    st.markdown("To apply this filter to a real sound, we convert \( H(f) \) back to the **time domain** using the inverse Fourier transform:")
    st.latex(r"h[n] = \text{IFFT}(H(f))")

    st.markdown("---")
    st.markdown("### 🔊 Final Step: Convolution")
    st.markdown("We apply the filter by convolving the real audio signal \( x[n] \) with the impulse response \( h[n] \):")
    st.latex(r"y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]")

    st.markdown("This results in a new sound 𝑦[𝑛]y[n]: the original audio, transformed to simulate how the other person would perceive it.")




# import soundfile as sf
# import numpy as np

# # Charger le fichier mono
# data, samplerate = sf.read(audio_path)  # data: (N,)

# # Vérifier que c'est bien mono
# if data.ndim == 1:
#     # Dupliquer le canal gauche pour créer un signal stéréo
#     stereo_data = np.stack([data, data], axis=1)  # shape: (N, 2)

#     # Sauvegarder en stéréo
#     sf.write("stereo.wav", stereo_data, samplerate)
#     print("Fichier stéréo créé avec succès.")
# else:
#     print("Ce fichier est déjà stéréo.")
