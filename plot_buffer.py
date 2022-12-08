import os, sys, time, argparse
import pyaudio
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

def main():

    savePathBuffer = "./buffer.wav"
    buffer_sec = 15
    update_sec = 0.01
    PlotBuffer(savePathBuffer, update_sec)

def loadBuffer(savePathBuffer):
    time.sleep(0.00001) # animate function did not work without this line!
    buffer = AudioSegment.from_wav(savePathBuffer)
    buffer = np.array(buffer.get_array_of_samples()) / 32767.0
    return buffer

def PlotBuffer(savePathBuffer, update_sec):
    buffer = loadBuffer(savePathBuffer)
    fig = plt.figure()
    ax1 = plt.axes(xlim=(0, len(buffer)), ylim=(-1.1,1.1))
    line, = ax1.plot([], [], lw=2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plotlays, plotcols = [2], ["black","red","blue","green"]
    lines = []
    for index in range(1):
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
            buffer = loadBuffer(savePathBuffer)
            size = buffer.shape[0]
            X = list(range(size))

            xlist = [X]
            ylist = [buffer]
            for lnum,line in enumerate(lines): line.set_data(xlist[lnum], ylist[lnum])
            print("here3")
        except:
            pass
        return lines

    ani = animation.FuncAnimation(fig, PlotBufferAnim, np.arange(1, len(buffer)), init_func=init,
    interval=interval, blit=False)
    plt.legend(["Audio Input"])
    # plt.xlim(0, 1000)
    # plt.ylim(-1, 1)
    plt.show()


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--annotsFolder', '-a', help="path to folder contatining annotations files", default="")
    # parser.add_argument('--wavsFolder', '-w', help="path to folder containing wav files", default="")
    # parser.add_argument('--outFolder', '-o', help="path to folder containing output files", default="")
    
    args = parser.parse_args()
    Flag = False
    # if args.annotsFolder == "": Flag = True
    # if args.wavsFolder == "":  Flag = True
    # if args.outFolder == "":   Flag = True
    if Flag:
        parser.print_help()
    else:
        main()#args.wavsFolder, args.annotsFolder, args.outFolder

