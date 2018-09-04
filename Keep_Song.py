
#from third-party
import numpy as np
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread

import shutil

#Spectrogram parameters
#Default windowing function in spectrogram function
#window =('tukey', 0.25) 
window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"

#Threshold for segmentation
threshold=2.0e6

# Contains the labels of the syllables from a single .wav file
song_y_n = ""


# Initial default for smooth_win = 2
def smooth_data(rawsong, samp_freq, freq_cutoffs=None, smooth_win=10):

    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)

    squared_song = np.power(filtsong, 2)

    #plt.figure(5)
    #x=np.arange(squared_song.shape[-1])
    #plt.plot(x, squared_song)
    #plt.show()
	
    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    #print("offset: %d" % offset)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return (smooth)
	
def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=(500, 10000)):
    """filter song audio with band pass filter, run through filtfilt
    (zero-phase filter)

    Parameters
    ----------
    rawsong : ndarray
        audio
    samp_freq : int
        sampling frequency
    freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter.
        If None, no cutoffs; filtering is done with cutoffs set
        to range from 0 to the Nyquist rate.
        Default is [500, 10000].

    Returns
    -------
    filtsong : ndarray
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError('Low frequency cutoff {} is invalid, '
                         'must be greater than zero.'
                         .format(freq_cutoffs[0]))

    Nyquist_rate = samp_freq / 2
    if freq_cutoffs[1] >= Nyquist_rate:
        raise ValueError('High frequency cutoff {} is invalid, '
                         'must be less than Nyquist rate, {}.'
                         .format(freq_cutoffs[1], Nyquist_rate))

    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    return filtsong
	
	
def segment_song(amp,
                 segment_params={'threshold': 5000, 'min_syl_dur': 0.2, 'min_silent_dur': 0.02},
                 time_bins=None,
                 samp_freq=None):
    """Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Parameters
    ----------
    amp : 1-d numpy array
        Either amplitude of power spectral density, returned by compute_amp,
        or smoothed amplitude of filtered audio, returned by evfuncs.smooth_data
    segment_params : dict
        with the following keys
            threshold : int
                value above which amplitude is considered part of a segment. default is 5000.
            min_syl_dur : float
                minimum duration of a segment. default is 0.02, i.e. 20 ms.
            min_silent_dur : float
                minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
    time_bins : 1-d numpy array
        time in s, must be same length as log amp. Returned by Spectrogram.make.
    samp_freq : int
        sampling frequency

    Returns
    -------
    onsets : 1-d numpy array
    offsets : 1-d numpy array
        arrays of onsets and offsets of segments.

    So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
    To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """

    if time_bins is None and samp_freq is None:
        raise ValueError('Values needed for either time_bins or samp_freq parameters '
                         'needed to segment song.')
    if time_bins is not None and samp_freq is not None:
        raise ValueError('Can only use one of time_bins or samp_freq to segment song, '
                         'but values were passed for both parameters')

    if time_bins is not None:
        if amp.shape[-1] != time_bins.shape[-1]:
            raise ValueError('if using time_bins, '
                             'amp and time_bins must have same length')

    above_th = amp > segment_params['threshold']
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    if time_bins is not None:
        # if amp was taken from time_bins using compute_amp
        # note that np.where calls np.nonzero which returns a tuple
        # but numpy "knows" to use this tuple to index into time_bins
        onsets = time_bins[np.where(above_th_convoluted > 0)]
        offsets = time_bins[np.where(above_th_convoluted < 0)]
    elif samp_freq is not None:
        # if amp was taken from smoothed audio using smooth_data
        # here, need to get the array out of the tuple returned by np.where
        # **also note we avoid converting from samples to s
        # until *after* we find segments** 
        onsets = np.where(above_th_convoluted > 0)[0]
        offsets = np.where(above_th_convoluted < 0)[0]

    if onsets.shape[0] < 1 or offsets.shape[0] < 1:
        return None, None  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1]  # duration of silent gaps
    if samp_freq is not None:
        # need to convert to s
        silent_gap_durs = silent_gap_durs / samp_freq
    keep_these = np.nonzero(silent_gap_durs > segment_params['min_silent_dur'])
    onsets = np.concatenate(
        (onsets[0, np.newaxis], onsets[1:][keep_these]))
    offsets = np.concatenate(
        (offsets[:-1][keep_these], offsets[-1, np.newaxis]))

    # eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    if samp_freq is not None:
        syl_durs = syl_durs / samp_freq
    keep_these = np.nonzero(syl_durs > segment_params['min_syl_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]

    if samp_freq is not None:
        onsets = onsets / samp_freq
        offsets = offsets / samp_freq

    return onsets, offsets


	
class Ask_Song(Thread):

    def __init__(self):
        Thread.__init__(self)

    def run(self):
        global song_y_n  
        song_y_n = input("Is there any song? Type y or n ")

pwd = os.getcwd()

os.chdir("../../../Documents/Recordings/Song/Song_Recorded_wav")
#retval = os.getcwd()
#print("Current working directory %s" % retval)	

#Take all files from the directory
songfiles_list = glob.glob('*.wav')	
	
#file_num is the index of the file in the songfiles_list
for file_num, songfile in enumerate(songfiles_list):

    #Read song file	
    print('File name %s' % songfile)
    (fs, rawsong) = wavfile.read(songfile)
    #print("Name of the songfile: %s" % songfile[0:16])
    rawsong = rawsong.astype(float)
	
	#Bandpass filter, square and lowpass filter
	#cutoffs : 1000, 8000
    amp = smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000))
		
	#Normalization of amp: Advantage: get rid of modulations of sound amplitude due to location of the bird in the cage. 
	#Disadvantage: if presence of stong call, the useful syllables are attenuated compared with syllables in song files with no calls
    #max_amp = np.amax(abs(amp))
    #print("max_rawsong1: %f" % max_rawsong)
    #amp = amp/max_amp
	
	#Segment song
    #(onsets, offsets) = segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': 0.02, 'min_silent_dur': 0.005},samp_freq=fs)
    #shpe = len(onsets)

    ########################################################################################
    # Create thread for indicating whether there is song or not in the file
    thread_1 = Ask_Song()
    # Start thread
    thread_1.start()
    ########################################################################################
	
    #Plot smoothed amplitude
    plt.figure() 
    x=np.arange(len(amp))
    plt.plot(x,amp)
	
    # #Plot onsets and offsets
    # for i in range(0,shpe):
    #     plt.axvline(x=onsets[i]*fs)
    #     plt.axvline(x=offsets[i]*fs,color='r')
    
	#Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, mode='complex')
    #sp_p=np.clip(abs(sp), 0, 0.004)
    max_sp=np.amax(abs(sp))
    plt.figure()
    sp = sp/max_sp
    #plt.imshow(abs(sp_p), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.show()

    #Wait for the labeling thread to finish
    thread_1.join()

    #If there is song in file, move the files txt and wav to an other location, else delete it
    if (song_y_n == "y"):
       Path_to_new_location_wav = "../Only_Song_Recorded_wav/"+songfile
       shutil.move(songfile, Path_to_new_location_wav)
       songfile_txt = songfile[0:15]+'.txt'
       Path_to_new_location_txt = "../Only_Song_Recorded_txt/"+songfile_txt
       Path_to_actual_location_txt = "../Song_Recorded_txt/"+songfile_txt
       shutil.move(Path_to_actual_location_txt, Path_to_new_location_txt)
    else:
       songfile_txt = songfile[0:15]+'.txt'
       Path_to_actual_location_txt = "../Song_Recorded_txt/"+songfile_txt
       os.remove(Path_to_actual_location_txt)
       Path_to_actual_location_wav = "../Song_Recorded_wav/"+songfile
       os.remove(Path_to_actual_location_wav)

       
#os.chdir(pwd)
