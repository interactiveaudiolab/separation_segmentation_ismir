import numpy as np
import librosa
from sklearn.decomposition import NMF
from scipy.spatial.distance import cosine

def find_template(music_stft, sr, min_t, n_components, start, end):
	template_stft = music_stft[:, start:end]
	layer = librosa.istft(template_stft)
	layer_rms = np.sqrt(np.mean(layer*layer))

	comps = []
	acts = []
	errors = []
	
	for T in range(min_t, n_components):
		transformer = NMF(n_components = T)
		comps.append(transformer.fit_transform(np.abs(template_stft)))
		acts.append(transformer.components_)
		errors.append(transformer.reconstruction_err_)
	

	#knee = np.diff(errors, 2)
	#knee = knee.argmax() + 2
	knee = 0
	#print 'Using %d components' % (knee + min_t)
	return comps[knee], acts[knee]

def find_template_knee(music_stft, sr, min_t, n_components, start, end):
	template_stft = music_stft[:, start:end]
	layer = librosa.istft(template_stft)
	layer_rms = np.sqrt(np.mean(layer*layer))

	comps = []
	acts = []
	errors = []
	
	for T in range(min_t, n_components):
		transformer = NMF(n_components = T)
		comps.append(transformer.fit_transform(np.abs(template_stft)))
		acts.append(transformer.components_)
		errors.append(transformer.reconstruction_err_)
	

	knee = np.diff(errors, 2)
	knee = knee.argmax() + 2
	print 'Using %d components' % (knee + min_t)
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

	#template - extracted template, residual - everything that's leftover.
	template = np.multiply(music_stft, mask)
	residual = np.multiply(music_stft, 1 - mask)

	return template, residual

def extract_reconstruction_errors(comps, music_stft, window_length, hop):
	K = comps.shape[1]
	#initialize transformer (non-negative matrix factorization) with K components
	transformer = NMF(n_components = K, init = 'custom')
	#W and H are random at first
	W = np.random.rand(comps.shape[0], K)
	start = 0
	errors = []

	while (start + window_length < music_stft.shape[1]):
		block = music_stft[:, start:start+window_length]
		
		H = np.random.rand(K, block.shape[1])
		W[:, 0:K] = comps
		
		params = {'W': W, 'H': H, 'update_W': False}
		comps_block = transformer.fit_transform(np.abs(block), **params)
		acts_block = transformer.components_
	
		#reconstruct the signal
		block_reconstruction = comps_block.dot(acts_block)
		errors.append(transformer.reconstruction_err_)

		start = start + hop
	return errors

from numpy.linalg import norm

def extract_reconstruction_error_beats(comps, music_stft, beats):
	K = comps.shape[1]
	#initialize transformer (non-negative matrix factorization) with K components
	transformer = NMF(n_components = K, init = 'custom')
	#W and H are random at first
	W = np.random.rand(comps.shape[0], K)
	start = 0
	errors = []
	lookback = 0
	weight = np.array([1 for i in range(2, music_stft.shape[0] + 2)])
	weight = weight/np.max(weight)
	for i in range(lookback+1, len(beats)):
		block = music_stft[:, beats[i-(lookback+1)]:beats[i]]
		
		H = np.random.rand(K, block.shape[1])
		W[:, 0:K] = comps
		
		params = {'W': W, 'H': H, 'update_W': False}
		comps_block = transformer.fit_transform(np.abs(block), **params)
		acts_block = transformer.components_

		#reconstruct the signal
		block_reconstruction = comps_block.dot(acts_block)
		
		block_reconstruction = block_reconstruction.T*weight
		block = block.T*weight
		distance = norm(block_reconstruction - np.abs(block))
		#errors.append(transformer.reconstruction_err_)
		errors.append(distance)
	return errors

def get_beat_spectrum(music):
	audio = AudioSignal(audio_data_array = music)
	r = Repet(audio_signal = audio, repet_type = RepetType.DEFAULT)
	r.run()
	b = r.get_beat_spectrum()
	b = np.square(b/b[0])
	return b

def get_beats(beat_spectrum, music, sr, rate):
	length_in_seconds = float(len(music))/sr
	conversion = length_in_seconds/len(beat_spectrum)

	auto_cosine = np.zeros((len(beat_spectrum), 1))
	for i in range(0, len(beat_spectrum) - 1):
		auto_cosine[i] = 1 - cosine(beat_spectrum[0:len(beat_spectrum) - i], beat_spectrum[i:len(beat_spectrum)])
	ac = auto_cosine[0:np.floor(auto_cosine.shape[0])/2]
	auto_cosine = np.vstack([ac[1], ac, ac[-2]])
	auto_cosine_diff = np.ediff1d(auto_cosine)
	sign_changes = auto_cosine_diff[0:-1]*auto_cosine_diff[1:]
	sign_changes = np.where(sign_changes < 0)[0]
	
	extrema_values = ac[sign_changes]

	e1 = np.insert(extrema_values, 0, extrema_values[0])
	e2 = np.insert(extrema_values, -1, extrema_values[-1])
	
	extrema_neighbors = np.stack((e1[0:-1], e2[1:]))
	
	m = np.amax(extrema_neighbors, axis=0)
	extrema_values = extrema_values.flatten() 
	maxima = np.where(extrema_values >= m)[0]
	maxima = zip(sign_changes[maxima], extrema_values[maxima])
	maxima = maxima[1:]
	maxima = sorted(maxima, key = lambda x: -x[1])
	period = maxima[0][0]
	beats = np.arange(0, length_in_seconds, (period/float(rate))*conversion)
	return period, beats

def window_rms(a, window_size):
	a2 = np.power(a, 2)
	window = np.ones(window_size)/float(window_size)
	return np.sqrt(np.convolve(a2, window, 'valid'))
