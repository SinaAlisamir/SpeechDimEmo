import torch
import torch.nn as nn
from torch.autograd import Variable
from pydub import AudioSegment
from python_speech_features import logfbank
import numpy as np
import os
# from Models import FeatureModel

class SER_Module(object):
    """
    SER module.
    Example:
        
    """
    def __init__(self, modelPath="", device="cpu", featModelPath=""):
        self.modelPath = modelPath
        self.device = device
        self.model = torch.load(self.modelPath, map_location=device)
        # print(os.path.split(os.path.split(self.modelPath)[0])[-1])
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

    def getEmo(self, audioPath):
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