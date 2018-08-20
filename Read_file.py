
#from third-party
import numpy as np
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread


# Useful global variables
#amp = evfuncs.smooth_data(self.rawAudio, self.sampFreq, self.spectParams['freq_cutoffs'])
#onsets, offsets = segment_song(amp, segment_params, samp_freq=self.sampFreq)	

#Spectrogram parameters
#window =('tukey', 0.25)
window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"
#colormap = None

#Threshold for segmentation
threshold=2.0e6

# COntains the lables of the syllables from a single .wav file
labels = []


def load_cbin(filename,channel=0):
    """
    loads .cbin files output by EvTAF. 
    
    arguments
    ---------
    filename : string

    channel : integer
        default is 0

    returns
    -------
    data : numpy array
        1-d vector of 16-bit signed integers

    sample_freq : integer
        sampling frequency in Hz. Typically 32000.
    """
    
    # .cbin files are big endian, 16 bit signed int, hence dtype=">i2" below
    data = np.fromfile(filename,dtype=">i2")
    recfile = filename[:-5] + '.rec'
    rec_dict = readrecf(recfile)
    data = data[channel::rec_dict['num_channels']]  # step by number of channels
    sample_freq = rec_dict['sample_freq']
    return data, sample_freq


def load_notmat(filename):
    """
    loads .not.mat files created by evsonganaly.m.
    wrapper around scipy.io.loadmat.
    Calls loadmat with squeeze_me=True to remove extra dimensions from arrays
    that loadmat parser sometimes adds.
    
    Argument
    --------
    filename : string, name of .not.mat file
     
    Returns
    -------
    notmat_dict : dictionary of variables from .not.mat files
    """

    if ".not.mat" in filename:
        pass
    elif filename[-4:] == "cbin":
            filename += ".not.mat"
    else:
        raise ValueError("Filename should have extension .cbin.not.mat or"
                         " .cbin")

    return loadmat(filename, squeeze_me=True)

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
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth

	
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

	
class Ask_Labels(Thread):

    def __init__(self, nb_segmented_syls):
        Thread.__init__(self)
        self.nb_segmented_syls = nb_segmented_syls

    def run(self):
        global labels  
        labels = []
        print("Total of %d segmented syllables" % self.nb_segmented_syls)
        for i in range (0, self.nb_segmented_syls):
            label = input("Label ?")
            #print("Type of label %s" % type(label))
            label = int(label)
            #print("Type of label %s" % type(label))
            labels.append(label)

pwd = os.getcwd()

os.chdir("..\SyllablesClassification\Koumura Data Set\Song_Data\March_07_nFB_morning_annot")
#retval = os.getcwd()
#print("Current working directory %s" % retval)	

songfiles_list = glob.glob('*.txt')	
	

	
#file_num is the index of the file in the songfiles_list
for file_num, songfile in enumerate(songfiles_list):	

    # Write file with onsets, offsets, labels
    current_dir = os.getcwd()
    file_path = current_dir+'\\'+songfile
    onsets = []
    offsets = []
    labels = [] 

    with open(file_path) as f:
         lines = f.readlines()
         for line in lines:
             #line = str(lines)
             print("line: %s" % line)
             splt = line.split(",")
             onsets.append(float(splt[0]))
             offsets.append(float(splt[1]))
             labels.append(str(splt[2]))

             splt_bis = (splt[2]).split("\n")
			 
             print("type of onsets %s " % type(float(splt[0])))
             print("%f %f %s" % (float(splt[0]),float(splt[1]),str(splt_bis[0])))
	
    f.close









