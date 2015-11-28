import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import sys
import time
from scipy.ndimage.filters import median_filter
from dtw import dtw
import os

music, sr = librosa.load(sys.argv[1])
original = np.copy(music)
original_rms = np.sqrt(np.mean(original*original))

music_stft = librosa.stft(music)
chroma = librosa.feature.chroma_cqt(y = music, sr = sr)
boundary_frames = librosa.segment.agglomerative(chroma, 20)
boundary_times = librosa.frames_to_time(boundary_frames)
bframes = []
i = 0
while i < len(boundary_times) - 1:
	previous_i = i
	for j in range(i+1, len(boundary_times)):
		if boundary_times[j] - boundary_times[i] > 3:
			bframes.append(boundary_times[j])
			i = j
			break
	if previous_i == i:
		break
boundary_frames = librosa.time_to_frames(bframes) - 1
print boundary_frames

start = time.time()

max_iterations = 10
max_T = 10

directory = sys.argv[1].split('/')[-1].split('.')

if not os.path.exists(directory[0]):
	os.makedirs(directory[0])
	os.makedirs(directory[0] + '/final')
	os.makedirs(directory[0] + '/templates')
	os.makedirs(directory[0] + '/residuals')
	os.makedirs(directory[0] + '/layers')

print 'Saving output to directory: %s\n' % directory[0]
librosa.output.write_wav(directory[0] + '/original.' + directory[-1], original, sr = sr)
directory = directory[0]

templates = []
residuals = []
layers = []

def find_template(music, sr, n_components, start, end):
	print 'Extracting template from frames %d to %d' % (start, end)
	template_stft = music_stft[:, start:end]
	librosa.output.write_wav(directory + '/layers/layer-' + str(i) + '.wav', librosa.istft(template_stft), sr = sr)
	layer = librosa.istft(template_stft)
	layer_rms = np.sqrt(np.mean(layer*layer))
	print 'Layer rms: ' + str(layer_rms)

	layers.append(layer)
	comps = []
	acts = []
	errors = []
	min_t = 2
	
	for T in range(min_t, n_components):
		transformer = NMF(n_components = T)
		comps.append(transformer.fit_transform(np.abs(template_stft)))
		acts.append(transformer.components_)
		errors.append(transformer.reconstruction_err_)
	
	knee = np.diff(errors, 2)
	knee = knee.argmax() + 2
	print 'Describing layer with %d components' % (knee + min_t)
	return comps[knee], acts[knee], knee + min_t, False

for i in range(max_iterations):
	if i >= len(boundary_frames):
		print 'Number of iterations exceeded boundary detections! Stopping extraction.\n'
		break
	print '\nIteration %d' % i
	music_stft = librosa.stft(music)
	if i == 0:
		previous_boundary = 0
	else:
		previous_boundary = boundary_frames[i-1]
	boundary = boundary_frames[i]

	comps, acts, K, bad_layer = find_template(music, sr, max_T, previous_boundary, boundary)

	if bad_layer:
		print 'Identified layer for extraction was bad! Stopping extraction.\n'
		break

	transformer = NMF(n_components = K, init = 'custom')
	
	W = np.random.rand(comps.shape[0], K)
	H = np.random.rand(K, music_stft.shape[1])
	#H[0:K, 0:acts.shape[1]] = acts
	W[:, 0:K] = comps
	params = {'W': W, 'H': H, 'update_W': False}
	comps_music = transformer.fit_transform(np.abs(music_stft), **params)
	acts_music = transformer.components_
	
	music_reconstruction = comps_music.dot(acts_music)
	music_stft_max = np.maximum(music_reconstruction, np.abs(music_stft))
	mask = np.divide(music_reconstruction, music_stft_max)
	mask = np.nan_to_num(mask)
	mask = np.round(mask)
	median_filter(mask, output = mask, size = 5)
	
	residual = np.multiply(music_stft, 1 - mask)
	template_stft = np.multiply(music_stft, mask)

	
	
	music = librosa.istft(template_stft)
	templates.append(music)
	librosa.output.write_wav(directory + '/templates/template-' + str(i) + '.wav', music, sr = sr)
	
	music = librosa.istft(residual)
	residuals.append(music)
	librosa.output.write_wav(directory + '/residuals/residual-' + str(i) + '.wav', music, sr = sr)
	
	rms = np.sqrt(np.mean(music*music))
	print 'Residual rms: ' + str(rms)
	if rms < original_rms/10:
		print 'Residual signal is basically empty. Stopping extraction.\n'
		break

print 'Combining similar templates\n'
onset_similarities = np.empty((len(layers), len(layers)))
mfcc_similarities = np.empty((len(layers), len(layers)))
layer_similarities = np.empty((len(layers), len(layers)))
layer_onsets = []
layer_mfccs = []
for l in layers:
	layer_onsets.append(librosa.onset.onset_strength(y = l, sr = sr))
	layer_mfccs.append(librosa.feature.mfcc(y = l, sr = sr))



for i in range(len(layers)):
	print 'Checking layer ' + str(i)
	for j in range(i+1, len(layers)):
		onset_similarities[i,j] = dtw(layer_onsets[i], layer_onsets[j])[0]
		mfcc_similarities[i,j] = dtw(layer_mfccs[i].T, layer_mfccs[j].T)[0]

W = 0
layer_similarities = W*mfcc_similarities/np.max(mfcc_similarities) + (1-W)*onset_similarities/np.max(onset_similarities)

upper_indices = np.triu_indices(len(layers), k = 1)
similarity_threshold = np.mean(layer_similarities[upper_indices])
combine = layer_similarities < .2
final_templates = range(len(templates))
combos = []

for i in zip(*upper_indices):
	if combine[i]:
		combos.append(i)		

for i in combos:
	if i[0] in final_templates and i[1] in final_templates:
		librosa.output.write_wav(directory + '/final/template-' + str(i[0]) + '_' + str(i[1]) + '.wav', templates[i[0]] + templates[i[1]], sr = sr)
		try:
			final_templates.remove(i[0])
			final_templates.remove(i[1])
		except:
			pass

print 'Writing out final sources'
for f in final_templates:
	librosa.output.write_wav(directory + '/final/template-' + str(f) + '.wav', templates[f], sr = sr)

print str(time.time() - start) + ' seconds elapsed.'
