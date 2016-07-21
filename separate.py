"""
Copyright {2016} {Prem Seetharaman, Bryan Pardo}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from functions import *
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.signal import correlate
from scipy.stats import skew

def load_file(path):
	music, sr = librosa.load(path)
	return music, sr, librosa.stft(music)

def segment_into_beats(music, sr):
	#librosa beat track
	tempo, beats = librosa.beat.beat_track(music)
	more_beats = []
	interpolation_factor = 2
	for i in range(0, len(beats) - 1):
		diff = np.floor((beats[i+1] - beats[i])/interpolation_factor)
		for j in range(0, interpolation_factor):
			more_beats.append(beats[i] + j*diff)
	more_beats.append(beats[-1])
	beats = np.array(more_beats)
	
	#or beat spectrum beat track
	#b = get_beat_spectrum(music, sr)
	#beats = get_beats(b, music, sr, 2)[1]
	#beats = librosa.time_to_frames(beats)
	music_stft = librosa.stft(music)
	return music_stft, beats

def quantize_track(music, sr):
    quantization = np.arange(0, len(music)/float(sr), .1)
    quantization = [int(b) for b in librosa.time_to_frames(quantization)]
    return quantization

def beat_sync_error(music_stft, beats, sr, n_components):
	comps, acts = find_template(music_stft, sr, n_components, n_components + 1, beats[0], beats[7])
	errors = extract_reconstruction_error_beats(comps, music_stft, beats)
	return errors

def get_template_error(music_stft, beats, comps):
	errors = extract_reconstruction_error_beats(comps, music_stft, beats)
	return errors

from scipy.signal import wiener

def mad(data):
	return np.median(np.abs(data - np.median(data)))

def find_inflection_point(errors, beats, music_stft, start, parameters):
	errors = np.square(errors)
	max_diff = np.max(np.abs(np.diff(errors, n=1)))
	lag = parameters['lag']
	p = parameters['p']
	q = parameters['q']
	means = []
	stds = []
	candidates = []
	d = lag
	while d < len(errors):
		#weighted = [np.array(errors[d-i]) for i in range(1, lag)]
		
		window = errors[d-lag:d-1]
		if np.abs(errors[d] - np.median(window)) > p*mad(window) and np.abs(errors[d] - errors[d-1]) > q*max_diff:
			candidates.append(d)
			d = d + lag
		else:
			d = d + 1
		'''
		if not means:
			means.append(np.mean(weighted))
			stds.append(np.std(weighted))
		if (np.abs(errors[d] - np.max(weighted))) > diff*stds[-1] and errors[d] >= threshold:
			print d, errors[d], threshold, means[-1], stds[-1]
			means.append((means[-1] + influence*errors[d])/(1 + influence))
			stds.append((stds[-1] + influence*np.sqrt((errors[d] - means[-1])**2))/(1 + influence))
			candidates.append(d)
		else:
			means.append(np.mean(weighted))
			stds.append(np.std(weighted))
		'''
	if not candidates:
		candidates = [len(errors) - 1]
	peak = candidates[0]
	#print 'Going to beat %d - %f seconds' % (peak + start, librosa.frames_to_time(beats[peak])[0])
	return beats[peak] - 4, peak + start
	

from scipy.spatial.distance import cosine

def frame_similarity(i, j, beats, music_stft):
	beat_i = np.abs(music_stft[:,beats[i]:beats[i+1]])
	beat_j = np.abs(music_stft[:,beats[j]:beats[j+1]])
	beat_i = np.max(beat_i, axis=1)
	#beat_i = beat_i/np.max(beat_i)
	beat_j = np.max(beat_j, axis=1)
	#beat_j = beat_j/np.max(beat_j)
	
	#c = correlate(beat_i, beat_j, mode='full')
	#return np.abs((np.argmax(c) - len(c)/2)/(float(len(c))/2))
	return cosine(beat_i, beat_j)

def frame_similarity_matrix(music_stft, beats):
	matrix = np.zeros((len(beats), len(beats)))
	for i in range(0, len(beats) - 1):
		for j in range(i, len(beats) - 1):
			matrix[i][j] = frame_similarity(i, j, beats, music_stft)
	return (matrix + matrix.T)/2

def find_first_beat_with_activity(music, sr, beats, start):
	beats = librosa.frames_to_time(beats)
	music = music[int(beats[int(start)]*sr):]
	total_rms = np.sqrt(np.mean(music*music))
	for i, b in enumerate(beats[start:]):
		sig = music[int(beats[i]*sr):int(beats[i+1]*sr)]
		rms = np.sqrt(np.mean(sig*sig))
		#print i, b, rms, total_rms/20
		if rms > total_rms/20:
			break
	return i + start

def get_layer(music, sr, start, beats=None, parameters=None, n_components = 8):
	if beats is not None:
		music_stft = librosa.stft(music)
	else:
		music_stft, beats = segment_into_beats(music, sr)
        beats = [int(b) for b in beats]
	if parameters is None:
		parameters = {'p': 5.5, 'q': .25, 'lag': 16}
	start = find_first_beat_with_activity(music, sr, beats, start)
	errors = beat_sync_error(music_stft, beats[start:], sr, n_components)
	inflection_point, beat = find_inflection_point(errors, beats[start:], music_stft, start, parameters)
	print 'Going from beat %d to %d' % (start, beat)
	comps, acts = find_template(music_stft, sr, n_components, n_components + 1, beats[start], inflection_point)
	template, residual = extract_template(comps, music_stft)
	template_error = get_template_error(music_stft, beats, comps)
	return template, residual, errors, beats, inflection_point, beat, template_error, start

def extract_all_layers(music_path, parameters=None, n_components = 8, beats = None):
	music, sr, music_stft = load_file(music_path)
	if beats == 'quantize':
		beats = quantize_track(music, sr)
	original_rms = np.sqrt(np.mean(music*music))
	layers = []
	boundaries = []
	template, residual, errors, beats, inflection_point, beat, template_error, start = get_layer(music, sr, 0, beats=beats, parameters=parameters, n_components=n_components)
	while True:
		layers.append(librosa.istft(template))
		boundaries.append(beats[beat])
		print 'LAYER: ' + str(len(layers))
		if np.sqrt(np.mean(music*music)) < original_rms/5:
			print 'Residual rms too low, terminating'
			break
		if beat >= len(beats)-8:
			print 'Went to end of file, terminating'
			break
		music = librosa.istft(residual)
		start = beat
		template, residual, errors, beats, inflection_point, beat, template_error, start = get_layer(music, sr, start, beats, parameters=parameters)
	return layers, boundaries, music_stft
