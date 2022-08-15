import PySimpleGUI as sg
import pyaudio
import wave
import numpy as np

""" RealTime PyAudio wave plot & FFT Transform graphical EQ Display"""

# VARS CONSTS:
_VARS = {'window': False,
         'stream': False,
         'audioData': np.array([])}

# pysimpleGUI INIT:
AppFont = 'Any 16'
sg.theme('DarkBlue2')
CanvasSizeWH = 500

layout = [[sg.Graph(canvas_size=(CanvasSizeWH, CanvasSizeWH),
                    graph_bottom_left=(-16, -16),
                    graph_top_right=(116, 116),
                    background_color='#1A2835',
                    key='graph')],
          [sg.ProgressBar(4000, orientation='h',
                          size=(20, 20), key='-PROG-')],
          [sg.Button('Listen', font=AppFont),
           sg.Button('Stop', font=AppFont, disabled=True),
           sg.Button('Exit', font=AppFont)]]
_VARS['window'] = sg.Window('Realtime PyAudio EQ Display',
                            layout, finalize=True)

graph = _VARS['window']['graph']

# INIT vars:
CHUNK = 128  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz
INTERVAL = 1  # Sampling Interval in Seconds ie Interval to listen
TIMEOUT = 10  # In ms for the event loop
GAIN = 0.6

wf = wave.open('Python_Playground\song_visualizer\Christopher Tin - The Storm-Driven Sea.wav', 'rb')
pAud = pyaudio.PyAudio()


# FUNCTIONS:
def drawAxis():
    graph.DrawLine((0, 1), (101, 1), color='#809AB6')  # Y Axis
    graph.DrawLine((1, 0), (1, 101), color='#809AB6')  # X Axis


def drawTicks():
    pad = 1
    divisionsX = 10
    multi = int(RATE/divisionsX)
    offsetX = int(100/divisionsX)

    divisionsY = 10
    offsetY = int(100/divisionsY)

    # ( x ➡️ , y ⬆️  ) Coordiante reference
    for x in range(0, divisionsX+1):
        # print('x:', x)
        graph.DrawLine(((x*offsetX)+pad, -3), ((x*offsetX)+pad, 3),
                       color='#809AB6')
        graph.DrawText(int((x*multi/1000)), ((x*offsetX), -6), color='#809AB6')

    for y in range(0, divisionsY+1):
        graph.DrawLine((-3, (y*offsetY)+pad), (3, (y*offsetY)+pad),
                       color='#809AB6')


def drawAxesLabels():
    graph.DrawText('kHz', (-10, 0), color='#809AB6')
    graph.DrawText('Freq. Level - Amplitude', (-5, 50),
                   color='#809AB6', angle=90)


def drawEQ():
    fft_data = np.fft.rfft(_VARS['audioData'])
    fft_data = np.absolute(fft_data)
    # print(fft_data)

    # attenuate first BIN:
    fft_data[0:8] = fft_data[0:8]/20
    
    # Calculate BINS (one way, there are others):
    # Take consecutive slices of 7 and sum values, resulting in 10 BINS..see:
    # https://stackoverflow.com/questions/29391815/sum-slices-of-consecutive-values-in-a-numpy-array
    BINS = [sum(fft_data[current: current+7])
            for current in range(0, len(fft_data), 7)]
    # Convert to numpy array:
    BINS = np.array(BINS)
    # Normalize and round up to values between 0-10
    BINS = np.interp(BINS, (BINS.min(), BINS.max()), (0, 10))
    BINS = np.round(BINS)    

    # Make Bars
    barStep = 10  # Height, width of bars
    pad = 2  # padding left,right,top,bottom
    for col, val in enumerate(BINS):
        # print('column:', int(col), ' gets ', int(val), 'Bars')
        for bar in range(0, int(val)):
            # print('bar', bar
            # ( x ➡️ , y ⬆️  ) Coordiante reference
            if bar < 3:
                barColor = '#00FF0E'
            elif bar < 6:
                barColor = 'yellow'
            elif bar < 9:
                barColor = 'orange'
            else:
                barColor = 'red'

            graph.draw_rectangle(top_left=((col*barStep)+pad, barStep*(bar+1)),
                                 bottom_right=((col*barStep)+barStep,
                                               (bar*barStep)+pad),
                                 line_color='black',
                                 line_width=2,
                                 fill_color=barColor)  # Conditional

# PYAUDIO STREAM :


def stop():
    if _VARS['stream']:
        _VARS['stream'].stop_stream()
        _VARS['stream'].close()
        _VARS['window']['-PROG-'].update(0)
        _VARS['window'].find_element('Stop').Update(disabled=True)
        _VARS['window'].find_element('Listen').Update(disabled=False)


def callback(in_data, frame_count, time_info, status):
    _VARS['audioData'] = np.frombuffer(in_data, dtype=np.int16)
    return (in_data, pyaudio.paContinue)


def listen():
    _VARS['window'].find_element('Stop').Update(disabled=False)
    _VARS['window'].find_element('Listen').Update(disabled=True)
    _VARS['stream'] = pAud.open(format=pAud.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True,
                                output_device_index = 3,
                                frames_per_buffer=CHUNK,
                                stream_callback=callback)
    _VARS['stream'].start_stream()


def updateUI():
    # Update volumne meter
    _VARS['window']['-PROG-'].update(np.amax(_VARS['audioData']))
    # Redraw :
    graph.erase()
    drawAxis()
    drawTicks()
    drawAxesLabels()    
    drawEQ()


# INIT:
drawAxis()
drawTicks()
drawAxesLabels()

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=TIMEOUT)
    if event == sg.WIN_CLOSED or event == 'Exit':
        stop()
        pAud.terminate()
        break

    if event == 'Listen':
        listen()

    if event == 'Stop':
        stop()

    elif _VARS['audioData'].size != 0:
        updateUI()


_VARS['window'].close()
wf.close()