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
import datetime
from datetime import datetime, timedelta

# AWS S3 Client setup
s3 = boto3.client('s3',
                  aws_access_key_id='',
                  aws_secret_access_key='',
                  config=Config(signature_version=UNSIGNED))

# S3 Bucket and file details
year = 2023
month = 10
filename = 'MARS-20231015T000000Z-2kHz.wav'
bucket = 'pacific-sound-2khz'
key = f'{year:04d}/{month:02d}/{filename}'
url = f'https://{bucket}.s3.amazonaws.com/{key}'

# Define the start time of the file and segment times
start_time = datetime.strptime("00:00:00", "%H:%M:%S")
segment_start_time = datetime.strptime("18:33:00", "%H:%M:%S")
segment_end_time = datetime.strptime("18:41:00", "%H:%M:%S")

# Calculate segment start and end in seconds
start_seconds = (segment_start_time - start_time).total_seconds()
end_seconds = (segment_end_time - start_time).total_seconds() + 59  # add 59 seconds to include the last minute

# Download and read data from S3
data, sample_rate = sf.read(io.BytesIO(urlopen(url).read()), dtype='float32')

# Calculate sample indices for the desired segment
start_sample = int(start_seconds * sample_rate)
end_sample = int(end_seconds * sample_rate)

# Ensure the segment is within the data range
if start_sample < len(data) and end_sample <= len(data):
    segment_data = data[start_sample:end_sample]
    
    # Optionally save the extracted audio segment
    output_path = 'MARS_1015183300.wav'
    sf.write(output_path, segment_data, sample_rate)
else:
    print("The requested time segment is out of the file's length bounds.")

