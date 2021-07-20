import collections
import signal
import sys
import time
import wave
import keras
import audio_utility as au
from array import array
from struct import pack

import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import webrtcvad

import librosa
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1000  # 1 sec jugement
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
# NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2

START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

vad = webrtcvad.Vad(1)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=CHUNK_SIZE)

got_a_sentence = False
leave = False

# initialize inference
feature_extractor = au.SpeechFeatures()
wanted_words = ['_silence_', '_unknown_', 'backward', 'forward', 'go', 'left', 'right', 'stop']
# import keras model
model = keras.models.load_model('models/vgg_19_model_mfcc_49x40_left_right_forward_backward_stop_go.hdf5')
model.summary()


def inference(test_audio):
    y, sr = librosa.load(test_audio, sr=16000)
    # fix the length of test audio to 1s or 16000 samples
    y = librosa.util.fix_length(y, size=16000)
    # normalize the test audio into [-1. 1]
    # y = np.clip(y, -1.0, 1.0)

    # extract the features
    audio_feature = feature_extractor.mfcc(y, fs=16000)
    audio_feature = audio_feature.flatten()
    audio_feature = np.expand_dims(audio_feature, axis=0)
    # inference
    prediction = model.predict(audio_feature)
    prediction = np.argmax(prediction)
    print(prediction, wanted_words[int(prediction)])


def handle_int(sig, chunk):
    global leave, got_a_sentence
    leave = True
    got_a_sentence = True


def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


signal.signal(signal.SIGINT, handle_int)

while not leave:
    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0

    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    buffer_in = ''
    # WangS
    raw_data = array('h')
    index = 0
    start_point = 0
    StartTime = time.time()
    stream.start_stream()

    while not got_a_sentence and not leave:
        chunk = stream.read(CHUNK_SIZE)
        # add WangS
        raw_data.extend(array('h', chunk))
        index += CHUNK_SIZE
        TimeUse = time.time() - StartTime

        active = vad.is_speech(chunk, RATE)

        # sys.stdout.write('1' if active else '_')
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index += 1
        ring_buffer_index %= NUM_WINDOW_CHUNKS

        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end += 1
        ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

        # start point detection
        if not triggered:
            ring_buffer.append(chunk)
            num_voiced = sum(ring_buffer_flags)
            if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                # sys.stdout.write(' Open ')
                triggered = True
                start_point = index - CHUNK_SIZE * 20  # start point
                # voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        # end point detection
        else:
            # voiced_frames.append(chunk)
            ring_buffer.append(chunk)
            num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
            if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or TimeUse > 10:
                triggered = False
                got_a_sentence = True

        sys.stdout.flush()

    sys.stdout.write('\n')

    stream.stop_stream()
    got_a_sentence = False

    # write to file
    raw_data.reverse()
    for index in range(start_point):
        raw_data.pop()
    raw_data.reverse()
    raw_data = normalize(raw_data)
    print("recorded")
    start = time.time()
    record_to_file("recording_.wav", raw_data, 2)
    print(f"time to save audio: {time.time() - start}")
    # leave = True
    start = time.time()
    inference("recording_.wav")
    print(f"time inference: {time.time() - start}")
    start = time.time()
    song = AudioSegment.from_wav("recording_.wav")
    play(song)
    print(f"time to play audio: {time.time() - start}")

stream.close()
