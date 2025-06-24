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

# Fonction pour générer un chirp exponentiel (balayage fréquentiel progressif)
def generate_chirp_exp(dur, freq_start, freq_end, Fs=44100):
    N = int(dur * Fs)  # Nombre d'échantillons
    t = np.arange(N) / Fs  # Axe temporel
    freq = np.exp(np.linspace(np.log(freq_start), np.log(freq_end), N))  # Évolution fréquentielle
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n-1] + 2 * np.pi * freq[n-1] / Fs  # Phase cumulée
    x = np.sin(phases)  # Signal audio final
    return x, t, freq

def read_audio_from_wav(audio_file):
    audio_file.seek(0)
    signal, sample_rate = sf.read(audio_file)
    return signal, sample_rate

# Enregistrement du signal audio dans un fichier temporaire .wav
def save_audio_to_wav(signal, sample_rate, filename="output.wav"):
    """Sauvegarde le signal en fichier WAV sans le normaliser"""
    signal_int16 = np.int16(signal)  # Conversion en format 16 bits
    write(filename, sample_rate, signal_int16)
    return filename

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

# Paramètres de génération du chirp
freq_start = 30
freq_end = 18000
dur = 10
Fs = 44100  # Fréquence d'échantillonnage

x, t, freq = generate_chirp_exp(dur, freq_start=freq_start, freq_end=freq_end, Fs=Fs)

# Affichage du spectrogramme du chirp pour vérification visuelle
fig, ax = plt.subplots(figsize=(7, 3))
N, H = 1024, 512
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N)  # Transformée de Fourier sur fenêtres
libfmp.b.plot_matrix(np.log(1 + np.abs(X)), Fs=Fs / H, Fs_F=N / Fs, ax=[ax], title='Spectrogram of chirp', colorbar=False)
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

MAX_INT16 = 32767 # Amplitude maximale pour un entier 16 bits
DURATION = 2

# Création de deux colonnes pour les deux utilisateurs
col1, col2 = st.columns(2)

# Bloc pour la première colonne d’interaction utilisateur
with col1:
    # Boucle sur chaque fréquence définie dans FREQUENCIES
    for i, freq in enumerate(FREQUENCIES):
        # Affiche la fréquence courante
        st.write(f"Frequency : {freq} Hz")

        # Crée un curseur pour permettre à l’utilisateur de régler l’amplitude du signal pour chaque fréquence
        amplitude_selectionnee = st.slider(
            f"Amplitude for {freq} Hz",  # Légende du slider
            min_value=0.0,               # Amplitude minimale (silence)
            max_value=1.0,               # Amplitude maximale (pleine échelle)
            value=0.5,                   # Valeur initiale par défaut
            step=0.01,                   # Pas du curseur
            key=f"col1_{i}",             # Clé unique pour éviter les conflits dans Streamlit
            help="Select the signal amplitude for this frequency"  # Info bulle
        )
        # Enregistre l’amplitude sélectionnée dans une liste
        amplitudes_selectionnees_col1.append(amplitude_selectionnee)

        # Génère un vecteur de temps pour la durée spécifiée
        temps = np.linspace(0, DURATION, int(Fs * DURATION), endpoint=False)

        # Génère un signal sinusoïdal avec la fréquence et l’amplitude choisies
        signal = amplitude_selectionnee * MAX_INT16 * np.sin(2 * np.pi * freq * temps)

        # Convertit le signal en fichier WAV temporaire
        audio_file = save_audio_to_wav(signal, Fs)

        # Affiche un lecteur audio pour écouter le signal généré
        st.audio(audio_file, format="audio/wav")

    # Convertit les amplitudes en dB pour l'affichage graphique (évite les erreurs log(0))
    amplitudes_selectionnees_col1_db = 20 * np.log10(np.array(amplitudes_selectionnees_col1) + 1e-12)

    # Vérifie si toutes les fréquences ont une amplitude sélectionnée
    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES):
        # Affiche le titre du graphique
        st.write("Amplitude versus frequency curve (Column 1)")

        # Crée le graphique Amplitude (en dB) vs Fréquence
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col1_db, marker='o', linestyle='-', color='b')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Amplitude versus frequency curve (Column 1)")
        plt.grid(True)

        # Affiche le graphique dans Streamlit
        st.pyplot(plt)
    else:
        # Message si l'utilisateur n’a pas sélectionné toutes les amplitudes
        st.write("Please select an amplitude for each frequency in column 1.")


# Bloc pour la deuxième colonne d’interaction utilisateur
with col2:
    # Boucle sur chaque fréquence définie dans FREQUENCIES
    for i, freq in enumerate(FREQUENCIES):
        # Affiche la fréquence courante
        st.write(f"Frequency : {freq} Hz")

        # Crée un curseur pour permettre à l’utilisateur de régler l’amplitude du signal pour chaque fréquence
        amplitude_selectionnee = st.slider(
            f"Amplitude for {freq} Hz",  # Légende du slider
            min_value=0.0,               # Amplitude minimale
            max_value=1.0,               # Amplitude maximale
            value=0.5,                   # Valeur initiale
            step=0.01,                   # Pas d’ajustement du slider
            key=f"col2_{i}",             # Clé unique pour éviter les conflits avec les sliders de col1
            help="Select the signal amplitude for this frequency"  # Info bulle
        )
        # Enregistre l’amplitude sélectionnée dans une liste
        amplitudes_selectionnees_col2.append(amplitude_selectionnee)

        # Génère un vecteur temps pour la durée spécifiée
        temps = np.linspace(0, DURATION, int(Fs * DURATION), endpoint=False)

        # Génère le signal sinusoïdal avec la fréquence et l’amplitude sélectionnées
        signal = amplitude_selectionnee * MAX_INT16 * np.sin(2 * np.pi * freq * temps)

        # Sauvegarde le signal sous forme de fichier audio WAV
        audio_file = save_audio_to_wav(signal, Fs)

        # Affiche un lecteur audio dans Streamlit pour écouter le signal
        st.audio(audio_file, format="audio/wav")

    # Convertit les amplitudes en dB pour le tracé du graphique
    amplitudes_selectionnees_col2_db = 20 * np.log10(np.array(amplitudes_selectionnees_col2) + 1e-12)

    # Vérifie que toutes les fréquences ont une amplitude sélectionnée
    if len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        # Affiche un titre descriptif du graphique
        st.write("Amplitude versus frequency curve (Column 2)")

        # Trace la courbe d’amplitude (en dB) en fonction de la fréquence
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, amplitudes_selectionnees_col2_db, marker='o', linestyle='-', color='r')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Amplitude versus frequency curve (Column 2)")
        plt.grid(True)

        # Affiche le graphique dans l’interface Streamlit
        st.pyplot(plt)
    else:
        # Affiche un message si toutes les amplitudes ne sont pas encore réglées
        st.write("Please select an amplitude for each frequency in column 2.")


# Bloc de traitement dans la première colonne (col1)
with col1:
    # Vérifie que l'utilisateur a sélectionné une amplitude pour chaque fréquence dans les deux colonnes
    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        
        # Calcul de la fonction de transfert en dB (différence entre colonne 2 et colonne 1)
        transfer_function_db1 = amplitudes_selectionnees_col2_db - amplitudes_selectionnees_col1_db
        
        # Affichage de la courbe de différence d’amplitude
        st.header("Amplitude difference to be applied (2-1)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, transfer_function_db1, marker='o', linestyle='-', color='g')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude Difference (dB)")
        plt.title("Amplitude difference to be applied (2-1)")
        plt.grid(True)
        st.pyplot(plt)

        # Titre pour la section de calcul IFFT (transformation inverse de Fourier)
        st.header("Inverse FFT")

        # Étape 1 : Convertir les valeurs de dB vers une échelle linéaire
        transfer_function_linear1 = 10**(transfer_function_db1 / 20)

        # Étape 2 : Créer un spectre symétrique pour simuler une réponse réelle
        # Partie positive du spectre (fréquences positives)
        positive_spectrum1 = transfer_function_linear1

        # Partie négative du spectre : miroir de la partie positive (symétrie Hermitienne)
        negative_spectrum1 = positive_spectrum1[::-1]  # Inverse les éléments

        # On ajoute une amplitude "1" pour la fréquence 0 Hz dans la partie positive
        positive_spectrum1 = np.insert(positive_spectrum1, 0, 1)

        # Préparation des fréquences pour l'affichage (axes X)
        negative_frequency1 = -np.array(FREQUENCIES[::-1])  # Fréquences négatives (miroir)
        positive_frequency1 = np.array(FREQUENCIES)
        positive_frequency1 = np.insert(positive_frequency1, 0, 0)  # Ajoute 0 Hz

        # Fusion des fréquences pour former un spectre double (X-axis complet)
        DOUBLE_FREQUENCIES = np.concatenate((negative_frequency1, positive_frequency1))

        # Fusion des amplitudes pour former le spectre symétrique complet (Y-axis)
        full_spectrum_linear1 = np.concatenate((negative_spectrum1, positive_spectrum1))

        # Affiche le spectre symétrique complet en échelle linéaire
        st.subheader("Linear Symmetric Spectrum")
        plt.figure(figsize=(8, 6))
        plt.plot(DOUBLE_FREQUENCIES, full_spectrum_linear1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Linear Symmetric Spectrum")
        plt.grid(True)
        st.pyplot(plt)

        # Étape 4 : Appliquer la transformation de Fourier inverse (IFFT)
        time_domain_response1 = np.fft.ifft(full_spectrum_linear1).real  # Prend uniquement la partie réelle

        # Conversion de la réponse temporelle en dB pour l’affichage
        time_domain_response_db1 = 20 * np.log10(np.abs(time_domain_response1) + 1e-12)  # On ajoute 1e-12 pour éviter log(0)

        # Affichage de la réponse impulsionnelle estimée
        st.subheader("Estimated Impulse Response")
        plt.figure(figsize=(8, 6))
        plt.plot(time_domain_response_db1)
        plt.xlabel("Time")
        plt.ylabel("Amplitude (dB)")
        plt.title("Impulse Response")
        plt.grid(True)
        st.pyplot(plt)



# Bloc de traitement dans la deuxième colonne (col2)
with col2:
    # Vérifie que l’utilisateur a sélectionné une amplitude pour chaque fréquence dans les deux colonnes
    if len(amplitudes_selectionnees_col1) == len(FREQUENCIES) and len(amplitudes_selectionnees_col2) == len(FREQUENCIES):
        
        # Calcul de la fonction de transfert inverse : colonne 1 - colonne 2
        transfer_function_db2 = amplitudes_selectionnees_col1_db - amplitudes_selectionnees_col2_db
        
        # Affichage de la différence d'amplitude (fonction de transfert) dans le sens inverse
        st.header("Amplitude difference to be applied (1-2)")
        plt.figure(figsize=(8, 6))
        plt.plot(FREQUENCIES, transfer_function_db2, marker='o', linestyle='-', color='g')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude Difference (dB)")
        plt.title("Amplitude difference to be applied (1-2)")
        plt.grid(True)
        st.pyplot(plt)

        # Titre pour la section de transformation de Fourier inverse
        st.header("Inverse FFT")

        # Étape 1 : Convertir la fonction de transfert de dB à une échelle linéaire
        transfer_function_linear2 = 10**(transfer_function_db2 / 20)

        # Étape 2 : Créer un spectre symétrique
        # Partie positive (fréquences mesurées)
        positive_spectrum2 = transfer_function_linear2

        # Partie négative (miroir de la partie positive)
        negative_spectrum2 = positive_spectrum2[::-1]

        # Ajout de la composante continue (0 Hz) avec un gain unitaire
        positive_spectrum2 = np.insert(positive_spectrum2, 0, 1)

        # Préparation des fréquences pour affichage (X-axis)
        negative_frequency2 = -np.array(FREQUENCIES[::-1])  # Fréquences négatives
        positive_frequency2 = np.array(FREQUENCIES)
        positive_frequency2 = np.insert(positive_frequency2, 0, 0)  # Inclut 0 Hz
        DOUBLE_FREQUENCIES = np.concatenate((negative_frequency2, positive_frequency2))

        # Construction du spectre complet en amplitude
        full_spectrum_linear2 = np.concatenate((negative_spectrum2, positive_spectrum2))

        # Affichage du spectre linéaire symétrique
        st.subheader("Linear Symmetric Spectrum")
        plt.figure(figsize=(8, 6))
        plt.plot(DOUBLE_FREQUENCIES, full_spectrum_linear2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Linear Symmetric Spectrum")
        plt.grid(True)
        st.pyplot(plt)

        # Étape 4 : Calcul de la réponse impulsionnelle via IFFT
        time_domain_response2 = np.fft.ifft(full_spectrum_linear2).real  # On garde uniquement la partie réelle

        # Conversion de la réponse temporelle en dB (ajout d’un epsilon pour éviter log(0))
        time_domain_response_db2 = 20 * np.log10(np.abs(time_domain_response2) + 1e-12)

        # Affichage de la réponse impulsionnelle estimée
        st.subheader("Estimated Impulse Response")
        plt.figure(figsize=(8, 6))
        plt.plot(time_domain_response_db2)
        plt.xlabel("Time")
        plt.ylabel("Amplitude (dB)")
        plt.title("Impulse Response")
        plt.grid(True)
        st.pyplot(plt)

# -----------------------------------------
# Affichage du son original importé par l'utilisateur
# -----------------------------------------

# Titre pour la section du son original
st.subheader("🔊 Original sound")

# Lecture de l’audio via Streamlit
st.audio(audio_path, format="audio/wav")

# Lecture des données audio avec scipy.io.wavfile
sample_rate, data = wavfile.read(audio_path)

# Séparation des deux canaux stéréo
left_channel = data[:, 0]  # Canal gauche
right_channel = data[:, 1]  # Canal droit


# Création de deux colonnes côte à côte
col3, col4 = st.columns(2)

# -----------------------------------
# Colonne 3 : Convolution et affichage du résultat pour la colonne 1
# -----------------------------------
with col3:
    st.subheader("🔊 Convolt result")

    # 1. Conversion des amplitudes en dB vers échelle linéaire
    amplitudes_lin = 10**(amplitudes_selectionnees_col1_db / 20)

    # 2. On ajoute une phase nulle (réelle, pas de partie imaginaire) pour la reconstruction du spectre
    spectre = amplitudes_lin

    # 3. Reconstruction du spectre complet (symétrie hermitienne) pour garantir un signal réel en temps
    # On prend le spectre et on y ajoute sa partie conjuguée inversée (sauf la première et dernière valeur)
    spectre_complet = np.concatenate([spectre, np.conj(spectre[-2:0:-1])])

    # 4. Calcul de la transformée de Fourier inverse (IFFT) pour obtenir le signal temporel
    signal_temps = np.fft.ifft(spectre_complet).real

    # 5. Convolution du signal original (gauche et droite) avec le signal temporel calculé
    convolved_left = fftconvolve(left_channel, signal_temps, mode="full")
    convolved_right = fftconvolve(right_channel, signal_temps, mode="full")

    # Recombinaison des canaux gauche et droite en un signal stéréo
    convolved11 = np.stack([convolved_left, convolved_right], axis=1)

    # Affichage graphique du signal convolué en dB
    plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(np.abs(convolved11)))
    plt.xlim(0, 10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    # Normalisation du signal convolué pour éviter la saturation audio
    convolved11 /= np.max(np.abs(convolved11) + 1e-6)

    # Sauvegarde en buffer mémoire au format WAV
    buffer1 = io.BytesIO()
    sf.write(buffer1, convolved11, sample_rate, format='WAV')

    # Lecture audio du résultat convolué
    st.audio(buffer1.getvalue(), format='audio/wav')

    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 2")

    # Convolution avec la réponse impulsionnelle estimée (time_domain_response1 calculée précédemment)
    convolved_left = fftconvolve(left_channel, time_domain_response1, mode="full")
    convolved_right = fftconvolve(right_channel, time_domain_response1, mode="full")

    # Recombinaison des canaux
    convolved1 = np.stack([convolved_left, convolved_right], axis=1)

    # Normalisation
    convolved1 /= np.max(np.abs(convolved1) + 1e-6)

    # Affichage graphique
    plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(np.abs(convolved1)))
    plt.xlim(0, 10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    # Sauvegarde et lecture audio
    buffer2 = io.BytesIO()
    sf.write(buffer2, convolved1, sample_rate, format='WAV')
    st.audio(buffer2.getvalue(), format='audio/wav')

# -----------------------------------
# Colonne 4 : Convolution et affichage du résultat pour la colonne 2
# -----------------------------------
with col4:
    st.subheader("🔊 Convolt result")

    # 1. Conversion dB → linéaire pour la colonne 2
    amplitudes_lin = 10**(amplitudes_selectionnees_col2_db / 20)

    # 2. Phase nulle ajoutée
    spectre = amplitudes_lin

    # 3. Reconstruction du spectre symétrique (Hermitien)
    spectre_complet = np.concatenate([spectre, np.conj(spectre[-2:0:-1])])

    # 4. Calcul IFFT → signal temporel
    signal_temps = np.fft.ifft(spectre_complet).real

    # 5. Convolution avec le signal original stéréo
    convolved_left = fftconvolve(left_channel, signal_temps, mode="full")
    convolved_right = fftconvolve(right_channel, signal_temps, mode="full")

    # Recombinaison stéréo
    convolved22 = np.stack([convolved_left, convolved_right], axis=1)

    # Affichage graphique du signal convolué
    plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(np.abs(convolved22)))
    plt.xlim(0, 10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 2")
    plt.grid(True)
    st.pyplot(plt)

    # Normalisation pour éviter saturation
    convolved22 /= np.max(np.abs(convolved22) + 1e-6)

    # Sauvegarde et lecture
    buffer1 = io.BytesIO()
    sf.write(buffer1, convolved22, sample_rate, format='WAV')
    st.audio(buffer1.getvalue(), format='audio/wav')

    st.subheader("🔊 Convolved result")
    st.write("Audio perceived as 1")

    # Convolution avec la réponse impulsionnelle estimée time_domain_response2
    convolved_left = fftconvolve(left_channel, time_domain_response2, mode="full")
    convolved_right = fftconvolve(right_channel, time_domain_response2, mode="full")

    # Recombinaison stéréo
    convolved2 = np.stack([convolved_left, convolved_right], axis=1)

    # Affichage graphique du signal convolué avec la réponse impulsionnelle
    plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(np.abs(convolved2)))
    plt.xlim(0, 10000)
    plt.xlabel("Time")
    plt.ylabel("Amplitude (dB)")
    plt.title("Convolved original sound with 1")
    plt.grid(True)
    st.pyplot(plt)

    # Normalisation et sauvegarde finale
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
