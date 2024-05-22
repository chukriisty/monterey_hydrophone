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
import librosa

s3 = boto3.client('s3',
    aws_access_key_id='',
    aws_secret_access_key='',
    config=Config(signature_version=UNSIGNED)) 

year = 2023
month = 10
filename = 'MARS-20231016T000000Z-2kHz.wav'
bucket = 'pacific-sound-2khz'
key = f'{year:04d}/{month:02d}/{filename}'

url = f'https://{bucket}.s3.amazonaws.com/{key}'

# read full-day of data
v, sample_rate = sf.read(io.BytesIO(urlopen(url).read()),dtype='float32')
v = v*3   # convert scaled voltage to volts

# Compute spectrogram
w = scipy.signal.get_window('hann',sample_rate)
f, t, psd = scipy.signal.spectrogram(v, sample_rate,nperseg=sample_rate,noverlap=0,window=w,nfft=sample_rate)
sens = -177.9  # hydrophone sensitivity at 250 Hz
np.seterr(divide='ignore')
psd = 10*np.log10(psd) - sens

start_hour = 20
start_minute = 33
start_sec = int(start_hour * 3600 + start_minute * 60 + 1)
end_sec = start_sec+60-1 #1 minute interval from 18:35:00


# # Subset the psd array using the indices
psd_subset = psd[:, start_sec:end_sec]

#Spectrum Level Plot
plt.figure(dpi=200, figsize = [5,2])
plt.imshow(psd_subset,aspect='auto',origin='lower',vmin=45,vmax=95)
plt.colorbar()
plt.ylim(8,1000)
plt.yscale('log')
plt.xlabel('Oct 15, 18:33:30 to 18:34:30')
plt.ylabel('Frequency (Hz)', fontsize=7)  
plt.title('Spectrum level (dB re 1 μμPa²/Hz)', fontsize=7)  # Corrected the title for clarity

# Adjust layout to prevent cutting off
plt.tight_layout() 
# plt.show()  # Display the plot
# Save the figure
plt.savefig(f'TEST_1016_20330.png')
plt.close()



# #Call Index Plot
# m = np.mean(psd_subset,axis=1)
# plt.figure(dpi=200, figsize = [7,3])
# plt.plot(m,f,'k')
# plt.ylim(650, 800)
# plt.xlim(30,150)
# # plt.plot([62, 81],[43, 43],'b--')
# # plt.plot([62, 81],[44, 44],'b--')
# # plt.plot([62, 81],[37, 37],'r--')
# # plt.plot([62, 81],[50, 50],'r--')
# plt.xlabel('Mean spectrum level (dB re 1 $\mu$Pa$^2$/Hz)')
# plt.ylabel('Frequency (Hz)')
# plt.show()

# #find the frequencies of the peak and average spectrum levels
# p1 = psd_subset[f==43]; p2 = psd_subset[f==44]
# pk = np.squeeze(np.array([p1,p2]))
# pk = np.mean(pk,axis=0); pk.shape
# # plt.plot(pk)
# # find the frequencies of the background and average
# b1 = psd_subset[f==37]; b2 = psd_subset[f==50]
# bg = np.squeeze(np.array([b1,b2]))
# bg = np.mean(bg,axis=0); bg.shape
# # plt.plot(pk/bg)
# # CI
# CI = pk/bg
# plt.figure(dpi=200, figsize = [9,3])
# plt.plot(CI)
# plt.xlabel('Seconds')
# plt.ylabel('CI')
# plt.show()
