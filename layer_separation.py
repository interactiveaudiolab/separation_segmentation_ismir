import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import sys
import time
from scipy.ndimage.filters import median_filter
from scipy.signal import wiener
from dtw import dtw
import os
import msaf

music, sr = librosa.load(sys.argv[1])
original = np.copy(music)
original_stft = librosa.stft(original)
original_rms = np.sqrt(np.mean(original*original))

music_stft = librosa.stft(music)

def get_boundaries(music_path):
	boundary_times, labels = msaf.process(music_path, feature="tonnetz", boundaries_id = 'cnmf', sonify_bounds=True)
	boundary_times = boundary_times[1:]
	bframes = boundary_times
	i = 0
	#while i < len(boundary_times) - 1:
	#	previous_i = i
	#	for j in range(i+1, len(boundary_times)):
	#		if boundary_times[j] - boundary_times[i] > 0:
	#			bframes.append(boundary_times[j])
	#			i = j
	#			break
	#	if previous_i == i:
	#		break
	print bframes
	boundary_frames = librosa.time_to_frames(bframes) - 1
	return boundary_frames

if len(sys.argv) <= 2:
	boundary_frames = get_boundaries(sys.argv[1])
else:
	boundary_frames = [float(x) for x in sys.argv[2:]]
	boundary_frames = librosa.time_to_frames(boundary_frames)

boundary_frames = np.append(boundary_frames, music_stft.shape[1])

print boundary_frames

start = time.time()

max_iterations = 20
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
layer_components = []

def find_template(music_stft, sr, n_components, start, end):
	print 'Extracting template from frames %d to %d' % (start, end)
	template_stft = music_stft[:, start:end]
	librosa.output.write_wav(directory + '/layers/layer-' + str(i) + '.wav', librosa.istft(music_stft[:, start:end]), sr = sr)
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
	layer_components.append(comps[knee])
	return comps[knee], acts[knee]

def extract_template(comps, music_stft):
	K = comps.shape[1]
	
	#initialize transformer (non-negative matrix factorization) with K components
	transformer = NMF(n_components = K, init = 'custom')
	
	#W and H are random at first
	W = np.random.rand(comps.shape[0], K)
	H = np.random.rand(K, music_stft.shape[1])
	
	#set W to be the template components you want to extract
	W[:, 0:K] = comps

	#don't let W get updated in the non-negative matrix factorization
	params = {'W': W, 'H': H, 'update_W': False}
	comps_music = transformer.fit_transform(np.abs(music_stft), **params)
	acts_music = transformer.components_
	
	#reconstruct the signal
	music_reconstruction = comps_music.dot(acts_music)

	#mask the input signal
	music_stft_max = np.maximum(music_reconstruction, np.abs(music_stft))
	mask = np.divide(music_reconstruction, music_stft_max)
	mask = np.nan_to_num(mask)
	
	#binary mask
	mask = np.round(mask)

	#median filter the mask to reduce noise
	#median_filter(mask, output = mask, size = 5)
	
	#template - extracted template, residual - everything that's leftover.
	template = np.multiply(original_stft, mask)
	residual = np.multiply(music_stft, 1 - mask)

	return template, residual

def extract_reconstruction_errors(comps, music_stft, window_length):
	print "Getting sliding reconstruction error"
	K = comps.shape[1]
	#initialize transformer (non-negative matrix factorization) with K components
	transformer = NMF(n_components = K, init = 'custom')
	#W and H are random at first
	W = np.random.rand(comps.shape[0], K)
	
	blocks = np.array_split(music_stft, np.floor(music_stft.shape[1]/window_length), axis = 1)
	errors = []
	
	for block_num, block in enumerate(blocks):
		H = np.random.rand(K, block.shape[1])
		W[:, 0:K] = comps

		params = {'W': W, 'H': H, 'update_W': False}
		comps_block = transformer.fit_transform(np.abs(block), **params)
		acts_block = transformer.components_
	
		#reconstruct the signal
		block_reconstruction = comps_block.dot(acts_block)
		errors.append(transformer.reconstruction_err_)
	return errors


def complete_signal(signal, comps):

	print 'Completing template'
		
	K = comps.shape[1]
	transformer = NMF(n_components = K, max_iter = 2)

	W = np.random.rand(comps.shape[0], K)
	H = np.random.rand(K, original_stft.shape[1])
	
	#set W to be the template components you want to extract
	W[:, 0:K] = comps

	params = {'W': W, 'H': H, 'update_W': False}
	comps_signal = transformer.fit_transform(np.abs(signal), **params)
	acts_signal = transformer.components_
	
	#reconstruct the signal
	signal_reconstruction = comps_signal.dot(acts_signal)

	#mask the input signal
	original_stft_max = np.maximum(signal_reconstruction, np.abs(original_stft))
	mask = np.divide(signal_reconstruction, original_stft_max)
	mask = np.nan_to_num(mask)
	
	#binary mask
	mask = np.round(mask)

	#median filter the mask to reduce noise
	median_filter(mask, output = mask, size = 5)
	
	#signal - extracted signal, residual - everything that's leftover.
	signal = np.multiply(original_stft, mask)
	return signal

def refine_template_iterate(template, residual, num_refinements, sr, max_T, start, end):
	for i in range(num_refinements):
		comps = find_template(template, sr, max_T, start, end)[0]
		template, r = extract_template(comps, template)
		residual += r
	return template, residual

def refine_template_wiener(music, template, residual):
	template = wiener(template)
	L = min(len(music), len(template))
	residual = music[0:L] - template[0:L]
	return template, residual

transformer = NMF(n_components = 50)
comps_original = transformer.fit_transform(np.abs(original_stft))
errors = []

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

	comps, acts = find_template(music_stft, sr, max_T, previous_boundary, boundary)
	template, residual = extract_template(comps, music_stft)
	#template, residual = refine_template_iterate(template, residual, 5, sr, max_T, previous_boundary, boundary)

	errors.append(extract_reconstruction_errors(comps, original_stft, window_length = 4))
	template = librosa.istft(template)
	#template, residual = refine_template_wiener(music, template, residual)
	templates.append(template)
	librosa.output.write_wav(directory + '/templates/template-' + str(i) + '.wav', template, sr = sr)
	
	residual = librosa.istft(residual)
	residuals.append(residual)
	librosa.output.write_wav(directory + '/residuals/residual-incomplete' + str(i) + '.wav', residual, sr = sr)
	
	residual = librosa.stft(residual)
	residual = complete_signal(residual, comps_original)
	residual = librosa.istft(residual)
	residuals.append(residual)
	librosa.output.write_wav(directory + '/residuals/residual-' + str(i) + '.wav', residual, sr = sr)
	music = residual
	

	rms = np.sqrt(np.mean(music*music))
	print 'Residual rms: ' + str(rms)
	if rms < original_rms/20:
		print 'Residual signal is basically empty. Stopping extraction.\n'
		break

fig = plt.figure()
for i, e in enumerate(errors):
	fig.add_subplot(len(errors), 1, i + 1)
	plt.plot(e)

plt.show()

'''
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
'''

print str(time.time() - start) + ' seconds elapsed.'
