import torch
import torch.nn as nn
from torch.autograd import Variable
from pydub import AudioSegment
from python_speech_features import logfbank
import numpy as np
import os
# from Models import FeatureModel

class VAD_Module(object):
    """
    VAD module to do voice activity detection on audio files.
    Example:
        path = "./test.wav"
        vad = VAD_Module()
        times = vad.timesFromFile(path)
        # This will return a list of times [start, end] (in seconds) of all the voices detected for the file at path, e.g. [[0.0, 3.39], [3.72, 5.48], [7.04, 7.87], [8.24, 8.65]].
    """
    def __init__(self, smoothedWin=0.5, mergeWin=0.5, modelPath="", device="cpu", hysteresis_bottom=-0.8, hysteresis_top=0, featModelPath=""):
        self.smoothedWin = smoothedWin
        self.modelPath = modelPath
        self.mergeWin = mergeWin*100
        self.device = device
        self.hysteresis_bottom = hysteresis_bottom
        self.hysteresis_top = hysteresis_top
        self.model = torch.load(self.modelPath, map_location=device)
        self.feat = os.path.split(os.path.split(self.modelPath)[0])[-1]
        if "W2V2" in self.feat or "wav2vec2" in self.feat: 
            self.getModelFeat(featModelPath, device=device)
    
    def getModelFeat(self, featModelPath, normalised=False, maxDur=29.98, device="cpu"):
        import fairseq
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([featModelPath])
        self.featModel = model[0]
        self.device = device
        self.featModel = self.featModel.to(device)
        self.featModel.eval()
        self.layerNormed = normalised
        self.maxDur = maxDur
    
    def timesFromFile(self, path):
        times = self.timesFromAudio(path)
        return times
    
    def timesFromAudio(self, audioPath):
        out = self.VadFromAudio(audioPath)
        times = self.getTimes(out)# Get the start and end times of each detected speech segment
        return times

    def VadFromAudio(self, audioPath):
        if "MFB" in self.feat:
            # print("fullPath", fullPath)
            inputs = self.getFeatsMFB(audioPath, winlen=0.025, winstep=0.01)
        if "W2V2" in self.feat or "wav2vec2" in self.feat:
            convOnly = False
            if "W2V2convOnly" in self.feat: convOnly=True
            self.layerNormed = False if "base" in self.feat else True
            inputs = self.getFeatsW2V2(audioPath, self.featModel, self.maxDur, self.layerNormed, device=self.device)
        if "standardized" in self.feat:
            inputs = (inputs - np.mean(inputs)) / np.std(inputs)
        feats = torch.FloatTensor(inputs)
        feats = Variable(feats).unsqueeze(0).to(self.device)
        out = self.model(feats)# Getting the output of the model based on input features
        out = out[:,:,:].view(out.size()[0]*out.size()[1]*out.size()[2])
        out = out.cpu().data.numpy()
        if self.smoothedWin>0: out = self.hysteresis(out, win=self.smoothedWin*100)# Smoothing the output of the model with a window of 0.5s (here it means 50 samples based on feature extraction process)
        out = self.mergeOuts(out)
        return out

    def getFeatsMFB(self, path, winlen=0.025, winstep=0.01):
        audio_file = AudioSegment.from_wav(path)
        sig = np.array(audio_file.get_array_of_samples())
        sig = sig / 32767.0 # scaling for 16-bit integer
        rate = audio_file.frame_rate
        fbank_feat = logfbank(sig, rate, winlen=winlen, winstep=winstep, nfilt=40, nfft=2028, lowfreq=0, highfreq=None, preemph=0.97) #, winfunc=np.hanning
        return fbank_feat

    def getFeatsW2V2(self, wavePath, model, maxDur, normalised, device="cpu"): # ADD "DICE" LATER
        # maxDur = 30
        # audio_file = AudioSegment.from_wav(path)
        audio_file = AudioSegment.from_wav(wavePath)
        rate = audio_file.frame_rate
        duration = audio_file.duration_seconds
        audio_files = []
        if duration < maxDur:
            audio_files.append(audio_file)
        while duration >= maxDur:
            audio_files.append(audio_file[:maxDur*1000])
            audio_file = audio_file[maxDur*1000:]
            duration = audio_file.duration_seconds
            if duration < maxDur: audio_files.append(audio_file)
        feats = []
        # n_cutts = len(audio_files)
        # audio_files.reverse()
        for audio_file in audio_files:
            sig = np.array(audio_file.get_array_of_samples())
            sig = sig / 32767.0 # scaling for 16-bit integer
            sigTensor = torch.from_numpy(sig).unsqueeze(0).float()
            sigTensor = sigTensor.to(device)
            if normalised:
                sigTensor = F.layer_norm(sigTensor, sigTensor.shape)
            c = model.extract_features(sigTensor, padding_mask=None, mask=False)[0]
            feat = c.squeeze(0).detach()
            feat = feat.to("cpu")
            feat = feat.numpy()
            # feat = np.append(feat, feat[-1]).reshape(feat.shape[0]+1, feat.shape[1]) # repeating the last item to match size!
            if len(feats)==0: 
                feats = feat
            else:
                feats = np.concatenate((feats, feat), axis=0)
            # print(feats.shape, sig.shape)
        return feats

    def getTimes(self, out, fs=0.01, duration_seconds=0):
        # if "W2V2" in self.feat or "wav2vec2" in self.feat: fs=0.02
        ins = []
        outs = []
        last = 0
        for i, o in enumerate(out):
            if o == 1 and last != 1: ins.append(i)
            if o == -1 and last == 1: outs.append(i)
            last = o
        if out[-1] == 1: outs.append(len(out)-1)
        times = []
        if duration_seconds != 0: fs = duration_seconds/len(out)
        for i, _ in enumerate(outs):
            times.append([round(ins[i]*fs,3), round(outs[i]*fs,3)])
        return times

    def smooth(self, sig, win=25*1):
        import numpy as np
        mysig = sig.copy()
        aux = int(win/2)
        for i in range(aux, len(mysig)-aux):
            value = np.mean(sig[i-aux:i+aux])
            mysig[i] = 1 if value > 0 else -1
        mysig[:aux] = 1 if np.mean(sig[:aux]) > 0 else -1
        mysig[-aux:] = 1 if np.mean(sig[-aux:]) > 0 else -1
        return mysig

    def hysteresis(self, sig, win=25*1):
        import numpy as np
        bottom = self.hysteresis_bottom
        top = self.hysteresis_top
        mysig = sig.copy()
        aux = int(win/2)
        mysig[0] = 1 if mysig[0] > top else -1
        for i in range(1, len(mysig)):
            if mysig[i] >= top:
                mysig[i] = 1
            if mysig[i] >= bottom and mysig[i] < top:
                if mysig[i-1] == 1: 
                    mysig[i] = 1
                else:
                    mysig[i] = -1
            if mysig[i] < bottom:
                mysig[i] = -1
        start = 0
        for i in range(1, len(mysig)):
            if mysig[i] == 1 and mysig[i-1] == -1: start = i
            if mysig[i] == -1 and mysig[i-1] == 1: 
                if i-start < win: 
                    for j in range(start, i):
                        mysig[j] = -1 
        return mysig

    def mergeOuts(self, out):
        myOut = out
        counter = 0; shouldCount = False; startC = 0
        for i in range(1, len(out)):
            if out[i-1] == 1 and out[i] == -1: 
                shouldCount = True
                startC = i
            if shouldCount:
                counter += 1
            if out[i-1] == -1 and out[i] == 1: 
                shouldCount = False
                if counter < self.mergeWin:
                    for j in range(startC, i):
                        myOut[j] = 1
                counter = 0
        return myOut
