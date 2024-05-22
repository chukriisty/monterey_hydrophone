import soundfile as sf
import numpy as np
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Read full-day of data
v, sample_rate = sf.read('../oct16_mars/wav_files/MARS_1016203300.wav')
v = v * 3  # Convert scaled voltage to volts

# Compute spectrogram to get the power spectral density (PSD)
n_fft = 4096  # Larger FFT size for better frequency resolution
hop_length = n_fft // 4  # Smaller hop length for better time resolution
w = scipy.signal.get_window('hann', n_fft)
f, t, psd = scipy.signal.spectrogram(v, sample_rate, nperseg=n_fft, noverlap=hop_length, window=w, nfft=n_fft)

# Hydrophone sensitivity at 250 Hz
sens = -177.9

# Convert PSD to power while considering hydrophone sensitivity
psd = 10 * np.log10(psd) - sens
power_spec = 10 ** (psd / 10)

# Convert power spectrogram to Mel spectrogram with the relevant frequency range
n_mels = 128
fmin = 20  # Minimum frequency for Mel bands
fmax = 10000  # Maximum frequency for Mel bands (Humpback whale calls can go up to 10 kHz)
mel_spectrogram = librosa.feature.melspectrogram(S=power_spec, sr=sample_rate, n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0, hop_length=hop_length)

# Apply PCEN with adjusted parameters
pcen_spec = librosa.pcen(mel_spectrogram, sr=sample_rate, hop_length=hop_length, gain=0.6, bias=10, power=0.25, time_constant=0.6)

# Select time slice
start_hour = 0
start_minute = 0
start_sec = int(start_hour * 3600 + start_minute * 60 + 1)
end_sec = start_sec + 300 - 1  # 5 minute interval

# Convert time slice to frame indices
start_frame = librosa.time_to_frames(start_sec, sr=sample_rate, hop_length=hop_length)
end_frame = librosa.time_to_frames(end_sec, sr=sample_rate, hop_length=hop_length)

# Subset the PCEN array
pcen_subset = pcen_spec[:, start_frame:end_frame]

# Plotting
plt.figure(figsize=(10, 4))
librosa.display.specshow(pcen_subset, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax, cmap='cool')
plt.colorbar()
plt.title('PCEN-enhanced Spectrogram of Humpback Whale Calls')

# Save the plot
plt.savefig(f'spectrogram_pcen_02.png')
plt.close()



# # Read full-day of data
# v, sample_rate = sf.read('../oct16_mars/wav_files/MARS_1016203300.wav')
# v = v * 3  # Convert scaled voltage to volts

# # Compute spectrogram
# w = scipy.signal.get_window('hann', sample_rate)
# f, t, psd = scipy.signal.spectrogram(v, sample_rate, nperseg=sample_rate, noverlap=0, window=w, nfft=sample_rate)
# sens = -177.9  # Hydrophone sensitivity at 250 Hz
# np.seterr(divide='ignore')
# psd = 10 * np.log10(psd) - sens

# # Select time slice
# start_hour = 00
# start_minute = 0
# start_sec = int(start_hour * 3600 + start_minute * 60 + 1)
# end_sec = start_sec + 300 - 1  # 5 minute interval

# # Subset the PSD array
# psd_subset = psd[:, start_sec:end_sec]

# # Convert dB to power
# power_spec = 10 ** (psd_subset / 10)


# S: np.ndarray (non-negative) The input (magnitude) spectrogram
# sr: number > 0 [scalar] The audio sampling rate
# hop_length: int > 0 [scalar] The hop length of S, expressed in samples
# gain: number >= 0 [scalar] The gain factor. Typical values should be slightly less than 1.
# bias: number >= 0 [scalar] The bias point of the nonlinear compression (default: 2)
# power: number >= 0 [scalar] The compression exponent. Typical values should be between 0 and 0.5. 
#     Smaller values of power result in stronger compression
# time_constant: number > 0 [scalar]
# eps: number > 0 [scalar] A small constant used to ensure numerical stability of the filter.
# b: number in [0, 1] [scalar] The filter coefficient for the low-pass filter.
# max_size: int > 0 [scalar] The width of the max filter applied to the frequency axis.
# ref: None or np.ndarray (shape=S.shape) An optional pre-computed reference spectrum 
# axis: int [scalar] The (time) axis of the input spectrogram
# max_axis: None or int [scalar]
# zi: np.ndarray The initial filter delay values. may be the zf (final delay values) of a previous call to pcen, or computed by scipy.signal.lfilter_zi


# Apply PCEN with adjusted parameters 
# Configurations: 
#128 bands between 8 Hz and 1Hz 
#Hann window of duration 128 ms and hop of 64
# s = 0.33, i.e. an averaging time scale of about T = 1 s
# pcen_spec = librosa.pcen(power_spec, sr=sample_rate, hop_length=64,
#                          gain=0.6, bias=10, power=0.25, time_constant=0.6)

# # Plotting
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(pcen_spec, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000, cmap='cool')
# plt.colorbar()
# plt.title('PCEN-enhanced Spectrogram of Humpback Whale Calls')

# # plt.show()
# plt.savefig(f'spectrogram_pcen{0:04}.png')
# plt.close()