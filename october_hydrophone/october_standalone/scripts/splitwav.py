from datetime import datetime, timedelta

# # Define the start time of the file
# start_time = datetime.strptime("17:59:09", "%H:%M:%S")

# # Define the segment start and end times
# segment_start_time = datetime.strptime("18:33:00", "%H:%M:%S")
# segment_end_time = datetime.strptime("18:41:00", "%H:%M:%S")

# # Calculate seconds from start_time to segment start and end times
# start_seconds = (segment_start_time - start_time).total_seconds()
# end_seconds = (segment_end_time - start_time).total_seconds() + 59 


import soundfile as sf

# Path to your large WAV file
file_path = '6715.231015175909.wav'

# Define the start time of the file and segment times
start_time = datetime.strptime("17:59:09", "%H:%M:%S")
segment_start_time = datetime.strptime("18:33:00", "%H:%M:%S")
segment_end_time = datetime.strptime("18:41:00", "%H:%M:%S")

# Calculate segment start and end in seconds
start_seconds = (segment_start_time - start_time).total_seconds()
end_seconds = (segment_end_time - start_time).total_seconds() + 59  # add 59 seconds to include the last minute

# Download and read data from S3
data, sample_rate = sf.read(file_path, dtype='float32')

# Calculate sample indices for the desired segment
start_sample = int(start_seconds * sample_rate)
end_sample = int(end_seconds * sample_rate)

# Ensure the segment is within the data range
if start_sample < len(data) and end_sample <= len(data):
    segment_data = data[start_sample:end_sample]
    
    # Optionally save the extracted audio segment
    output_path = 'TEST_STAND_1015183300.wav'
    sf.write(output_path, segment_data, sample_rate)
else:
    print("The requested time segment is out of the file's length bounds.")







# # Open the file to read sample rate and calculate the sample indices
# with sf.SoundFile(file_path) as sound_file:
#     sr = sound_file.samplerate
#     start_sample = int(start_seconds * sr)
#     end_sample = int(end_seconds * sr)
#     frames_to_read = end_sample - start_sample

#     # Ensure we don't read past the file length
#     if start_sample < len(sound_file) and start_sample + frames_to_read <= len(sound_file):
#         sound_file.seek(start_sample)
#         data = sound_file.read(frames=frames_to_read)

#         # Save the extracted audio to a new file
#         output_path = '../oct15_standalone/wav_files/STAND_1015183300.wav'
#         sf.write(output_path, data, sr)
#     else:
#         print("The requested time segment is out of the file's length bounds.")