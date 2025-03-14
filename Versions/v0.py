import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import libfmp.b

from streamlit_advanced_audio import (
    CustomizedRegion,
    RegionColorOptions,
    WaveSurferOptions,
    audix,
)

def temps_recherche(temps, vecteur_temps):
  """
  Retourne l'indice de la case du tableau de temps correspondant au temps donné.

  Args:
    temps: Le temps pour lequel on cherche l'indice.
    vecteur_temps: Le tableau NumPy contenant les valeurs de temps.

  Returns:
    L'indice de la case correspondant au temps donné, ou None si le temps
    n'est pas trouvé dans le tableau.
  """
def compute_equal_loudness_contour(freq_min, freq_max, num_points=100):
    """Computation of the equal loudness contour

    Notebook: C1/C1S3_Dynamics.ipynb

    Returns:
        equal_loudness_contour (np.ndarray): Equal loudness contour (in dB)
        freq_range (np.ndarray): Evaluated frequency points
    """
    freq_range = np.logspace(np.log10(min_freq), np.log10(max_freq), num=num_points)
    freq = 1000
    # Function D from https://bar.wikipedia.org/wiki/Datei:Acoustic_weighting_curves.svg
    h_freq = ((1037918.48 - freq**2)**2 + 1080768.16 * freq**2) / ((9837328 - freq**2)**2 + 11723776 * freq**2)
    n_freq = (freq / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq / ((freq**2 + 79919.29) * (freq**2 + 1345600)))
    h_freq_range = ((1037918.48 - freq_range**2)**2 + 1080768.16 * freq_range**2) / ((9837328 - freq_range**2)**2
                                                                                     + 11723776 * freq_range**2)
    n_freq_range = (freq_range / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq_range / ((freq_range**2 + 79919.29) *
                                                                         (freq_range**2 + 1345600)))
    equal_loudness_contour = 20 * np.log10(np.abs(n_freq / n_freq_range))
    return equal_loudness_contour, freq_range

    # Recherche de l'indice le plus proche du temps donné
    indice = np.argmin(np.abs(vecteur_temps - temps))

    return indice

def temps_recherche(current_time, t):
    return np.argmin(np.abs(t - current_time))

def generate_chirp_exp_equal_loudness(dur, freq_start, freq_end, Fs=22050):
    """Generation chirp with exponential frequency increase and equal loudness

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        dur (float): Length (seconds) of the signal
        freq_start (float): Starting frequency of the chirp
        freq_end (float): End frequency of the chirp
        Fs (scalar): Sampling rate (Default value = 22050)

    Returns:
        x (np.ndarray): Generated chirp signal
        t (np.ndarray): Time axis (in seconds)
        freq (np.ndarray): Instant frequency (in Hz)
        intensity (np.ndarray): Instant intensity of the signal
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    intensity, freq = compute_equal_loudness_contour(freq_min=freq_start, freq_max=freq_end, num_points=N)
    amp = 10**(intensity / 20)
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n-1] + 2 * np.pi * freq[n-1] / Fs
    x = amp * np.sin(phases)
    return x, t, freq, intensity



sample_rate=44100


# Initialize session state variables
if "tab0" not in st.session_state:
    st.session_state.tab0 = []
if "tab1" not in st.session_state:
    st.session_state.tab1 = []
if "current_time" not in st.session_state:
    st.session_state.current_time = None
if "f_save" not in st.session_state:
    st.session_state.f_save = None

t = np.linspace(0, 20, int(20 * 44100))
f = np.linspace(0, 11500, int(20 * 44100))
x = np.zeros_like(t)

for i in range(len(t)):
    x[i] = np.sin(2 * np.pi * f[i] * t[i])

st.subheader("Test fréquence minimum et maximum")
st.write("Écouter le son et sauvegarder la fréquence minimale entendue et la fréquence maximale entendue")

result2 = audix(
    data=x,
    sample_rate=sample_rate,
    wavesurfer_options=WaveSurferOptions(wave_color="#00b894", height=80),
)

if result2:
    st.session_state.current_time = result2["currentTime"]
    st.write(st.session_state.current_time)
    index = temps_recherche(st.session_state.current_time, t)
    st.session_state.f_save = f[index]
    st.write(st.session_state.f_save)

if st.button("Sauvegarder la fréquence"):
    st.session_state.tab0.append(st.session_state.current_time)
    st.session_state.tab1.append(st.session_state.f_save)

if st.button("Supprimer la dernière entrée"):
    if st.session_state.tab0:
        st.session_state.tab0.pop()
        st.session_state.tab1.pop()
        st.rerun()  # Use st.rerun()

if st.button("Effacer tout le tableau"):
    st.session_state.tab0 = []  # Réinitialiser la liste des temps
    st.session_state.tab1 = []  # Réinitialiser la liste des fréquences
    st.rerun()

df = pd.DataFrame(
    {
        "Temps": st.session_state.tab0,
        "Fréquences": st.session_state.tab1,
    }
)

st.table(df)

if st.button("Tracer la courbe isosonique"):
    min_freq=st.session_state.tab1[0]
    max_freq=st.session_state.tab1[1]
    equal_loudness_contour, freq_range = compute_equal_loudness_contour(min_freq, max_freq)
    fig,ax=plt.subplots()

    libfmp.b.plot_signal(equal_loudness_contour, T_coef=freq_range, figsize=(6,3), xlabel='Frequency (Hz)',
                        ylabel='Intensity (dB)', title='Equal loudness contour', color='red',ax=ax)
    
    ax.set_xscale('log')
    ax.grid()

    st.pyplot(fig)


# dur=10
# min_freq=st.session_state.tab1[0]
# max_freq=st.session_state.tab1[1]
# x_equal_loudness, t, freq, intensity = generate_chirp_exp_equal_loudness(dur, freq_start=min_freq, freq_end=max_freq, Fs=sample_rate)

# st.subheader("Volume sonore égal")
# st.audio(x_equal_loudness,sample_rate=sample_rate)

if len(st.session_state.tab1) >= 2:
    dur = 10
    min_freq = st.session_state.tab1[0]
    max_freq = st.session_state.tab1[1]

    x_equal_loudness, t, freq, intensity = generate_chirp_exp_equal_loudness(
        dur, freq_start=min_freq, freq_end=max_freq, Fs=sample_rate
    )

    st.subheader("Volume sonore égal")
    st.audio(x_equal_loudness, sample_rate=sample_rate)

