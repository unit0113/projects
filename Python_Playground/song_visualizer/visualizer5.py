import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys
import wave
import pygame


class AudioStream(object):
    def __init__(self):

        self.last_read = 0

        # Get song data
        wf = wave.open('Python_Playground\song_visualizer\Christopher Tin - The Storm-Driven Sea.wav', 'rb')

        self.sample_freq = wf.getframerate()
        self.num_samples = wf.getnframes()
        self.signal_wave = wf.readframes(-1)
        self.chunk = 2048
        wf.close()

        self.signal_array = np.frombuffer(self.signal_wave, dtype=np.int8)
        self.ft = np.abs(np.fft.fft((self.signal_array)))
        self.signal_array = (self.signal_array * 255 / np.max(self.signal_array) + 128) * 0.8
        
        # X axis linespace
        self.x = np.arange(0, 2 * self.chunk, 2)
        self.f = np.linspace(0, self.sample_freq / 2, self.chunk)

        # Initialize pyqtgraph
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 115, 1910, 1070)

        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '0'), (127, '128'), (255, '255')]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])

        sp_xlabels = [(np.log10(10), '10'), (np.log10(100), '100'), (np.log10(1000), '1000'), (np.log10(22050), '22050')]
        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])

        self.waveform = self.win.addPlot(title='WAVEFORM', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis})
        self.spectrum = self.win.addPlot(title='SPECTRUM', row=2, col=1, axisItems={'bottom': sp_xaxis})

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(0, 255, padding=0)
                self.waveform.setXRange(0, 2 * self.chunk, padding=0.005)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=3)
                self.spectrum.setLogMode(x=True, y=True)
                self.spectrum.setYRange(-4, 0, padding=0)
                self.spectrum.setXRange(np.log10(20), np.log10(self.sample_freq / 2), padding=0.005)

    def update(self):
        self.set_plotdata(name='waveform', data_x=self.x, data_y=self.signal_array[self.last_read:self.last_read+self.chunk])
        self.set_plotdata(name='spectrum', data_x=self.f, data_y=self.ft[self.last_read:self.last_read+self.chunk])
        self.last_read += self.chunk

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        pygame.mixer.init()
        pygame.mixer.music.load('Python_Playground\song_visualizer\Christopher Tin - The Storm-Driven Sea.mp3')
        pygame.mixer.music.play()
        self.start()


if __name__ == '__main__':

    audio_app = AudioStream()
    audio_app.animation()