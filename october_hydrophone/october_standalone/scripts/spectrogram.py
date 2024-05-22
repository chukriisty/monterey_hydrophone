import scipy
from scipy import signal
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Updated file path
split_file_path = '../oct15_standalone/wav_files/STAND_1015183300.wav'
v, sample_rate = sf.read(split_file_path, dtype='float32')
v = v * 3  # Convert scaled voltage to volts

# Compute spectrogram with updated parameters
nperseg = sample_rate // 6  # Shorter segments for better time resolution
noverlap = nperseg // 2     # 50% overlap for smoother spectrogram
nfft = sample_rate          # Higher FFT size for better frequency resolution
w = scipy.signal.get_window('hann', nperseg)
f, t, Sxx = scipy.signal.spectrogram(v, sample_rate, nperseg=nperseg, noverlap=noverlap, window=w, nfft=nfft)

# Convert to dB scale
Sxx = 10 * np.log10(Sxx + 1e-10)  # Adding a small value to avoid log(0)

start_sec = 420
end_sec = start_sec + 60  # 2-minute interval

# Convert time in seconds to the appropriate index in the spectrogram
start_idx = int(start_sec * (1 / (nperseg / sample_rate)))
end_idx = int(end_sec * (1 / (nperseg / sample_rate)))

# Subset the psd array using the indices
Sxx_subset = Sxx[:, start_idx:end_idx]

# Spectrum Level Plot
plt.figure(dpi=200, figsize=[8, 3])
plt.imshow(Sxx_subset, aspect='auto', origin='lower', vmin=-100, vmax=-50, extent=[start_sec, end_sec, f[0], f[-1]])
# plt.imshow(Sxx_subset, aspect='auto', origin='lower', vmin=-100, vmax=-50, extent=[start_sec, end_sec, f[0], f[-1]], cmap='viridis')
plt.colorbar(label='Power Spectral Density (dB)')
plt.ylim(8, 1000)
plt.yscale('log')
plt.xlabel('Oct 15 Standalone, 18:40:00 to 18:41:00', fontsize=10)
plt.ylabel('Frequency (Hz)', fontsize=10)
plt.title('Spectrum level (dB)', fontsize=10)

# Adjust layout to prevent cutting off
plt.tight_layout()
# Save the figure
plt.savefig(f'../oct15_standalone/spectrograms/1min_intervals/1015_184000.png')
plt.close()



# # Spectrum Level Plot
# plt.figure(dpi=200, figsize=[8, 3])
# plt.imshow(psd_subset, aspect='auto', origin='lower', vmin=45, vmax=95, extent=[start_sec, end_sec, f[0], f[-1]])
# plt.colorbar(label='Power Spectral Density (dB re 1 μPa²/Hz)')
# plt.ylim(8, 1000)
# plt.yscale('log')
# plt.xlabel('Oct 15 MARS, 18:34:27 to 18:36:27', fontsize=10)
# plt.ylabel('Frequency (Hz)', fontsize=10)
# plt.title('Spectrum level (dB re 1 μPa²/Hz)', fontsize=10)

# import soundfile as sf
# import matplotlib.pyplot as plt
# import scipy.signal
# import numpy as np
# import librosa

# # Read full-day of data
# v, sample_rate = sf.read('STAND_1016203300.wav')
# v = v * 3  # Convert scaled voltage to volts

# # Compute the entire spectrogram
# w = scipy.signal.get_window('hann', sample_rate)
# f, t, psd = scipy.signal.spectrogram(v, sample_rate, nperseg=sample_rate, noverlap=0, window=w, nfft=sample_rate)
# sens = -140.66  # Hydrophone sensitivity at 250 Hz
# np.seterr(divide='ignore')
# psd = 10 * np.log10(psd) - sens

# # Duration of each segment (5 minutes)
# segment_duration = 5 * 60  # 5 minutes in seconds
# num_segments = int(np.floor(t[-1] / segment_duration)) 

# # Generate and save a plot for each 5-minute interval
# for i in range(num_segments):
#     start_sec = i * segment_duration
#     end_sec = start_sec + segment_duration

#     # Ensure we do not go out of index bounds
#     if end_sec > t[-1]:
#         end_sec = int(t[-1])
    
#     start_idx = np.searchsorted(t, start_sec)
#     end_idx = np.searchsorted(t, end_sec)

#     # Subset the PSD array using the indices
#     psd_subset = psd[:, start_idx:end_idx]

#     # Create plot
#     plt.figure(dpi=200, figsize=[5, 2])
#     plt.imshow(psd_subset, aspect='auto', origin='lower', vmin=45, vmax=95)
#     plt.colorbar()
#     plt.ylim(8, 10000)
#     plt.yscale('log')
#     plt.xlabel(f'Time: {start_sec // 3600:02}:{(start_sec % 3600) // 60:02}:{start_sec % 60:02} to {end_sec // 3600:02}:{(end_sec % 3600) // 60:02}:{end_sec % 60:02}')
#     plt.ylabel('Frequency (Hz)', fontsize=7)
#     plt.title('Spectrum level (dB re 1 μPa²/Hz)', fontsize=7)

#     # Save the figure
#     plt.savefig(f'spectrogram_{i:04}.png')
#     plt.close()
