'''
Audio stream -> Buffer -> VAD -> SER -> Emotion
'''
from datetime import datetime
import os, sys, time, argparse
import pyaudio
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from VAD_Module import VAD_Module
from SER_Module import SER_Module
import speech_recognition

def main(sessionID):
    print("Running module ...")
    rate = 16000 # in Hertz
    chunk_sec = 1 # in Seconds
    buffer_sec = 15 # in Seconds
    program_time = 15*60 # in Seconds
    device = "cuda" # "cuda" or "cpu", Note: "cpu" should only be used when running MFB feature extraction for both VAD and SER modules
    saveFolder = os.path.join(".", "Sessions", sessionID)
    savePathBuffer = os.path.join(saveFolder, "buffer.wav") # where the buffer file would be stored
    savePathSession = os.path.join(saveFolder, "session.wav") # where the buffer file would be stored
    savePathVAD = os.path.join(saveFolder, "VAD_segments") # where VAD detected files would be stored
    savePathSER = os.path.join(saveFolder, "SER_segments") # where VAD detected files would be stored with a simplified tag of emotion

    VADmodelPath="./Models/Recola_46_MFB_standardized_GRU_32-1/model.pth" # where the pytorch model is stored for VAD # Recola_46_MFB_standardized_GRU_32-1
    SERmodelPath="./Models/valence_MFB_standardized_GRU_32-1_0/model.pth" # where the pytorch model is stored for SER # valence_MFB_standardized_GRU_32-1_0
    VADfeatModelPath = "/mnt/HD-Storage/Models/FlowBERT_2952h_base.pt" # where the wav2vec2 model is stored (in case of using wav2vec2 for feature extraction)
    SERfeatModelPath = "/mnt/HD-Storage/Models/wav2vec2.0_models/2.6K_base/24_06_2021/checkpoint_best.pt" # where the wav2vec2 model is stored (in case of using wav2vec2 for feature extraction)
    smoothedWin = 0.100 # in Seconds
    mergeWin = 0.75 # in Seconds
    hysteresis_bottom = -0.8 # Amplitude (between -1 and +1)
    hysteresis_top = 0.8 # Amplitude (between -1 and +1)

    VADModule = VAD_Module(smoothedWin=smoothedWin, mergeWin=mergeWin, modelPath=VADmodelPath, device=device, hysteresis_bottom=hysteresis_bottom, hysteresis_top=hysteresis_top, featModelPath=VADfeatModelPath)
    SERModule = SER_Module(modelPath=SERmodelPath, device=device, featModelPath=SERfeatModelPath)  
    
    
    p=pyaudio.PyAudio()
    CHUNK = int(chunk_sec*rate)#2**15
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=rate, input=True, frames_per_buffer=CHUNK)
    num_of_chunks = program_time/chunk_sec
    
    for i in range(int(num_of_chunks)):
        ########## BUFFER LOOP ##############
        # time0 = time.time()
        audioChunk = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        # time1 = time.time()
        overWrite=True if i==0 else False
        Buffer(audioChunk, savePathBuffer, buffer_sec, rate=rate, overWrite=overWrite) # Saves comming audio stream to buffer
        Buffer(audioChunk, savePathSession, program_time, rate=rate, overWrite=overWrite) # Saves comming audio stream to session (unlike buffer, session contains all the audio from the start of the program)
        # time2 = time.time()
        # print("Buffer time:", time2-time1)

        ########## VAD and SER LOOP ##############
        vadOuts = VAD(savePathBuffer, savePathSession, VADModule, device=device, perc=chunk_sec/buffer_sec, outputFolder=os.path.join(saveFolder, "VAD"))
        detected = saveDetected(savePathBuffer, vadOuts, savePathVAD, chunk_sec=chunk_sec, rate=rate)
        # time3 = time.time()
        # print("VAD time:", time3-time2)
        serOuts = SER(savePathBuffer, SERModule, device=device, perc=chunk_sec/buffer_sec, outputFolder=os.path.join(saveFolder, "SER"))
        # if detected: saveValence(savePathBuffer, savePathSER, detected, serOuts)
        if detected: 
            transcription, label = saveFileLabelled(savePathBuffer, savePathVAD, savePathSER, detected, serOuts, delayPerc=0/buffer_sec)
            if label == "neutral": print(str(datetime.now()), transcription, "😐")
            if label == "positive": print(str(datetime.now()), transcription, "🙂")
            if label == "negative": print(str(datetime.now()), transcription, "😞")
            if label == "positive2": print(str(datetime.now()), transcription, "😃")
            if label == "negative2": print(str(datetime.now()), transcription, "😠")
        # time4 = time.time()
        # print("SER time:", time4-time3)
        # print("Total time:", time4-time0)

def Buffer(audioChunk, savePathBuffer, buffer_sec, rate=16000, overWrite=False):
    try:
        lastBuffer = AudioSegment.from_wav(savePathBuffer)
        lastBufferArray = np.array(lastBuffer.get_array_of_samples())
        newChunk = np.concatenate((lastBufferArray, audioChunk), axis=0)
    except:
        newChunk = audioChunk
    if overWrite: newChunk = audioChunk
    # print("newChunk", len(newChunk)/rate, len(audioChunk)/rate)
    if len(newChunk)/rate > buffer_sec:
        newChunk = newChunk[len(newChunk)-buffer_sec*rate:]
    buffer = AudioSegment(
        newChunk.tobytes(), 
        frame_rate=rate,
        sample_width=newChunk.dtype.itemsize, 
        channels=1
    )
    if not os.path.exists(os.path.dirname(savePathBuffer)): os.makedirs(os.path.dirname(savePathBuffer))
    buffer.export(savePathBuffer, format="wav")
    # return buffer.duration_seconds

def VAD(savePathBuffer, savePathSession, VADModule, device="cpu", perc=2/15, outputFolder="."):
    session = AudioSegment.from_wav(savePathSession)
    duration_seconds = session.duration_seconds
    out = VADModule.VadFromAudio(savePathBuffer)
    outLng = len(out)
    outPath = os.path.join(outputFolder, 'VAD_outs.csv')
    outSessionPath = os.path.join(outputFolder, 'VAD_outs_session.csv')
    timesSessionPath = os.path.join(outputFolder, 'VAD_times_session.csv')
    try:
        lastOutSession = np.loadtxt(outSessionPath, dtype='float', delimiter=',', usecols=(0), unpack=True)
        lastOut = np.loadtxt(outPath, dtype='float', delimiter=',', usecols=(0), unpack=True)
        # print(len(out), -int(perc*len(out)))
        outSession = np.concatenate((lastOutSession, out[outLng-int(perc*outLng):]), axis=0)
        out = np.concatenate((lastOut, out[outLng-int(perc*outLng):]), axis=0)
    except:
        outSession = out
    if len(out) > outLng:
        out = out[len(out)-outLng:]
    # times = VADModule.getTimes(out)
    if not os.path.exists(os.path.dirname(outPath)): os.makedirs(os.path.dirname(outPath))
    np.savetxt(outPath, out, delimiter=",")
    np.savetxt(outSessionPath, outSession, delimiter=",")
    # outSessionTimes = np.interp(np.linspace(0, outSession.shape[0], int(BufferArray.shape[0]*0.001)), np.arange(0, len(outSession), 1), outSession)
    # print(outSession.shape, BufferArray.shape)
    times = VADModule.getTimes(outSession, duration_seconds=duration_seconds)
    np.savetxt(timesSessionPath, times, delimiter=",")
    return out
    # data = np.loadtxt('VAD_times.csv', dtype='str', delimiter=',', usecols=(0, 1), unpack=True)
    # print("VAD times", times)

def saveDetected(savePathBuffer, vadOuts, savePathVAD, chunk_sec=2, rate=16000):
    data, sr = sf.read(savePathBuffer)
    # print(vadOuts)
    endT = 0; strT = 0
    for i in range(len(vadOuts)-2, 0, -1):
        # if vadOuts[i+1] != vadOuts[i]: print(i, vadOuts[i+1], vadOuts[i])
        if vadOuts[i+1] == -1 and vadOuts[i] == 1: endT = i
        if vadOuts[i+1] == 1 and vadOuts[i] == -1: strT = i; break
    if endT == 0: return False
    strT = int(strT*len(data)/len(vadOuts))
    endT = int(endT*len(data)/len(vadOuts))
    if endT < len(data) - rate*chunk_sec: return False # only check last chunk for detected segment to avoid multiple detection of the same segment!
    segment = data[strT:endT]
    fileName = str(datetime.now())
    segmentPath = os.path.join(savePathVAD, fileName +'.wav')
    # segmentPath = os.path.join(self.savePath, originalName, fileName)
    # if outPath != "": segmentPath = os.path.join(outPath, originalName, fileName)
    if not os.path.exists(os.path.dirname(segmentPath)): os.makedirs(os.path.dirname(segmentPath))
    sf.write(segmentPath, segment, sr)
    return strT, endT, fileName

def SER(savePathBuffer, SERModule, device="cpu", perc=2/15, outputFolder="."):
    out = SERModule.getEmo(savePathBuffer)
    outLng = len(out)
    outPath = os.path.join(outputFolder, 'SER_outs.csv')
    outSessionPath = os.path.join(outputFolder, 'SER_outs_session.csv')
    timesSessionPath = os.path.join(outputFolder, 'SER_times_session.csv')
    try:
        lastOutSession = np.loadtxt(outSessionPath, dtype='float', delimiter=',', usecols=(0), unpack=True)
        lastOut = np.loadtxt(outPath, dtype='float', delimiter=',', usecols=(0), unpack=True)
        # print(len(out), -int(perc*len(out)))
        outSession = np.concatenate((lastOutSession, out[outLng-int(perc*outLng):]), axis=0)
        out = np.concatenate((lastOut, out[outLng-int(perc*outLng):]), axis=0)
    except:
        outSession = out
    if len(out) > outLng:
        out = out[len(out)-outLng:]
    # times = VADModule.getTimes(out)
    if not os.path.exists(os.path.dirname(outPath)): os.makedirs(os.path.dirname(outPath))
    np.savetxt(outPath, out, delimiter=",")
    np.savetxt(outSessionPath, outSession, delimiter=",")
    return out
    # data = np.loadtxt('VAD_times.csv', dtype='str', delimiter=',', usecols=(0, 1), unpack=True)
    # print("SER out", out)

def saveValence(savePathBuffer, savePathSER, detected, serOuts):
    strT, endT, fileName = detected
    segmentPath = os.path.join(savePathSER, fileName+'.csv')
    if not os.path.exists(os.path.dirname(segmentPath)): os.makedirs(os.path.dirname(segmentPath))
    np.savetxt(segmentPath, serOuts, delimiter=",")

def saveFileLabelled(savePathBuffer, savePathVAD, savePathSER, detected, serOuts, delayPerc=0):
    data, sr = sf.read(savePathBuffer)
    strT, endT, fileName = detected
    delay = int((delayPerc)*len(serOuts))
    serOutsSeg = serOuts[int(strT*len(serOuts)/len(data))+delay : int(endT*len(serOuts)/len(data))+delay]
    label = "neutral"
    target = np.mean(serOutsSeg)
    if target > 0.1: label = "positive"
    if target > 0.25: label = "positive2"
    if target < -0.1: label = "negative"
    if target < -0.2: label = "negative2"
    segmentPathVAD = os.path.join(savePathVAD, fileName+'.wav')
    segment, sr = sf.read(segmentPathVAD)
    r = speech_recognition.Recognizer() 
    with speech_recognition.AudioFile(segmentPathVAD) as source: 
        audio = r.record(source) 
        try:
            transcription = r.recognize_google(audio, language='fr-FR')
        except: 
            transcription = "UNK"
    segmentPathSER = os.path.join(savePathSER, fileName+"_"+transcription+"_"+label+"_"+str(round(target,3))+'.wav')
    if not os.path.exists(os.path.dirname(segmentPathSER)): os.makedirs(os.path.dirname(segmentPathSER))
    sf.write(segmentPathSER, segment, sr)
    return transcription, label

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessionID', '-s', help="ID of the session", default="Session_0")
    
    args = parser.parse_args()
    Flag = False
    if args.sessionID == "": Flag = True
    if Flag:
        parser.print_help()
    else:
        main(args.sessionID)

