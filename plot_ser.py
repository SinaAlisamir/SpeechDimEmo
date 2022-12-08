import os, sys, time, argparse
import pyaudio
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

def main(sessionID):
    saveFolder = os.path.join(".", "Sessions", sessionID)
    savePathBuffer = os.path.join(saveFolder, "buffer.wav") # where the buffer file would be stored
    SERFolder = os.path.join(saveFolder, "SER")
    update_sec = 0.01
    PlotSER(savePathBuffer, update_sec, SERFolder)

def loadBuffer(savePathBuffer):
    time.sleep(0.000001) # animate function did not work without this line!
    buffer = AudioSegment.from_wav(savePathBuffer)
    duration = buffer.duration_seconds
    print(buffer.duration_seconds)
    buffer = np.array(buffer.get_array_of_samples()) / 32767.0
    return buffer, duration

def loadSER(SERFolder):
    serOutsPath = os.path.join(SERFolder, 'SER_outs.csv')
    time.sleep(0.000001) # animate function did not work without this line!
    vadOuts = np.loadtxt(serOutsPath, dtype='float', delimiter=',', usecols=(0), unpack=True)
    return vadOuts

def PlotSER(savePathBuffer, update_sec, SERFolder):
    buffer, duration = loadBuffer(savePathBuffer)
    fig = plt.figure()
    ax1 = plt.axes(xlim=(-duration, 0), ylim=(-1.1,1.1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plotlays, plotcols = [2], ["black","red"]
    lines = []
    for index in range(2):
        lobj = ax1.plot([],[],lw=2,color=plotcols[index])[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    x1,y1 = [],[]

    interval = update_sec*1000

    def PlotBufferAnim(i):
        print("here1.5")
        try:
            buffer, duration = loadBuffer(savePathBuffer)
            vadOuts = loadSER(SERFolder)
            print(buffer.shape, vadOuts.shape)
            size = buffer.shape[0]
            vadOuts = np.interp(np.linspace(0, vadOuts.shape[0], size), np.arange(0, len(vadOuts), 1), vadOuts)
            print(buffer.shape, vadOuts.shape)
            # X = list(range(size))
            X = np.linspace(-duration, 0, size)

            xlist = [X, X]
            ylist = [buffer, vadOuts]
            for lnum,line in enumerate(lines): line.set_data(xlist[lnum], ylist[lnum])
            print("here3")
        except:
            pass
        return lines

    ani = animation.FuncAnimation(fig, PlotBufferAnim, np.arange(1, len(buffer)), init_func=init,
    interval=interval, blit=False)
    plt.legend(["Audio Input", "SER Output"])
    # plt.xlim(0, 1000)
    # plt.ylim(-1, 1)
    plt.show()


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

