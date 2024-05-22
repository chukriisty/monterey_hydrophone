import boto3, botocore
from botocore import UNSIGNED
from botocore.client import Config
from urllib.request import urlopen
import io
import scipy
from scipy import signal
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

split_file_path = '../oct15_mars/wav_files/MARS_1015183300.wav'
v, sample_rate = sf.read(split_file_path, dtype='float32')
v = v * 3   # convert scaled voltage to volts

# Compute spectrogram with a better balance of segment length and overlap
nperseg = sample_rate // 6  # Shorter segments for better time resolution
noverlap = nperseg // 2     # 50% overlap for smoother spectrogram
w = scipy.signal.get_window('hann', nperseg)
f, t, psd = scipy.signal.spectrogram(v, sample_rate, nperseg=nperseg, noverlap=noverlap, window=w, nfft=sample_rate)
sens = -177.9  # hydrophone sensitivity at 250 Hz
np.seterr(divide='ignore')
psd = 10 * np.log10(psd) - sens

start_sec = 420
end_sec = start_sec + 60  # 1-minute interval

# Convert time in seconds to the appropriate index in the spectrogram
start_idx = int(start_sec * (1 / (nperseg / sample_rate)))
end_idx = int(end_sec * (1 / (nperseg / sample_rate)))

# Subset the psd array using the indices
psd_subset = psd[:, start_idx:end_idx]

# Spectrum Level Plot
plt.figure(dpi=200, figsize=[8, 3])
plt.imshow(psd_subset, aspect='auto', origin='lower', vmin=45, vmax=95, extent=[start_sec, end_sec, f[0], f[-1]])
plt.colorbar(label='Power Spectral Density (dB re 1 μPa²/Hz)')
plt.ylim(8, 1000)
plt.yscale('log')
plt.xlabel('Oct 15 MARS, 18:40:00 to 18:41:00', fontsize=10)
plt.ylabel('Frequency (Hz)', fontsize=10)
plt.title('Spectrum level (dB re 1 μPa²/Hz)', fontsize=10)

# Adjust layout to prevent cutting off
plt.tight_layout()
# Save the figure
plt.savefig(f'../oct15_mars/spectrograms/1min_intervals/1015_184000.png')
plt.close()






# import boto3, botocore
# from botocore import UNSIGNED
# from botocore.client import Config
# from urllib.request import urlopen
# import io
# import scipy
# from scipy import signal
# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt


# split_file_path = '../oct15_mars/wav_files/MARS_1015183300.wav'
# v, sample_rate = sf.read(split_file_path, dtype='float32')
# v = v*3   # convert scaled voltage to volts

# # Compute spectrogram
# w = scipy.signal.get_window('hann',sample_rate)
# f, t, psd = scipy.signal.spectrogram(v, sample_rate,nperseg=sample_rate,noverlap=0,window=w,nfft=sample_rate)
# sens = -177.9  # hydrophone sensitivity at 250 Hz
# np.seterr(divide='ignore')
# psd = 10*np.log10(psd) - sens

# start_sec = 360
# end_sec = start_sec+120-1 #5 minute interval 

# # # Subset the psd array using the indices
# psd_subset = psd[:, start_sec:end_sec]

# #Spectrum Level Plot
# plt.figure(dpi=200, figsize = [5,3])
# plt.imshow(psd_subset,aspect='auto',origin='lower',vmin=45,vmax=95)
# plt.colorbar()
# plt.ylim(8,1000)
# plt.yscale('log')
# plt.xlabel('Oct 15 MARS, 18:39:00 to 18:41:00', fontsize=6)
# plt.ylabel('Frequency (Hz)', fontsize=6)  
# plt.title('Spectrum level (dB re 1 μμPa²/Hz)', fontsize=6)  # Corrected the title for clarity

# # Adjust layout to prevent cutting off
# plt.tight_layout() 
# # plt.show()  # Display the plot
# # Save the figure
# plt.savefig(f'../oct15_mars/spectrograms/5min_intervals/1015_183900.png')
# plt.close()
