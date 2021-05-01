import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
import requests
import numpy as np
import urllib


class StockData:

    """ Class to store stock data for requested tickers, takes in initial input from get_tickers
    """    
    
    def __init__(self, symbol):
        """ Takes in ticker from get_tickers output and initiates the class

        Args:
            symbol (str): Stocker ticker
        """ 

        self.symbol = symbol
        

def get_CIK_list():
    """ Webscrape list of stock tickers and their CIK numbers, ouput into a dictionary

    Returns:
        cikList (dict): dictionary of pairs of stock tickers and CIK numbers
    """

    # Initial data pull
    cikData = urllib.request.urlopen("https://www.sec.gov/include/ticker.txt")    
    if cikData.getcode() != requests.codes.ok:
        pass
        # messagebox error in GUI
    
    # Parse into dict
    cikList = {}
    for line in cikData:
        line = line.decode("utf-8")
        (key, val) = line.split()
        cikList[key] = val

    return cikList    



def get_tickers():
    """ Function to open up a GUI that allows the user to select options. Can input one ticker for single stock DD.
    Can input two tickers for comparision. Can select a flag that will create a CSV for inport into excel dashboard. Also a quit option to abort execution of program

    Returns:
        ticker1 (str): first ticker input
        ticker2 (str): second ticker input
        ticker3 (bool): excel export flag
    """    
    # GUI
    root = tk.Tk()
    root.title("Python EDGAR Scraper")

    # Size of GUI
    gui_width = 670
    gui_height = 590

    # Size of screen
    screenWidth = root.winfo_screenwidth()
    screenHeight = root.winfo_screenheight()

    # Place GUI in middle of screen
    x = (screenWidth / 2) - (gui_width / 2)
    y = (screenHeight / 2) - (gui_height / 2)
    root.geometry(f'{gui_width}x{gui_height}+{int(x)}+{int(y)}')

    # Function for exiting GUI
    def clickExitButton():
        sys.exit()

    # Delete contents of entry boxes when clicked on
    def entry_clear(e):
        if ticker1Entry.get() == "Ticker 1" or ticker2Entry.get() == "Ticker 2":
            ticker1Entry.delete(0, tk.END)
            ticker2Entry.delete(0, tk.END)
    
    # Pull ticker and execute rest of program
    def execute():
        # Pull tickers
        ticker1 = ticker1Entry.get()
        ticker2 = ticker2Entry.get()

        # Initial ticker check
        if ticker1 == ("" or "Ticker 1"):
            t1Flag = False
        else:
            t1Flag = True
        if ticker2 == ("" or "Ticker 2"):
            t2Flag = False
        else:
            t2Flag = True

        # If no data in one but data in two
        if t1Flag == False and t2Flag == True:
            ticker1 = ticker2
            ticker2 = ""
            t2Flag = False
            t1Flag = True

        # If no tickers
        if t1Flag == False and t2Flag == False:
            tk.messagebox.showwarning(title="No Tickers Provided", message="Please Provide at least one Stock ticker")

    # Set GUI size and layout
    canvas = tk.Canvas(root, width=gui_width, height=gui_height)
    canvas.pack(fill="both", expand="True")
    root.resizable(width=False, height=False)

    # Background
    bg = ImageTk.PhotoImage(file=".venv/programs/EDGAR/bg.png")
    canvas.create_image(0,0, image=bg, anchor="nw")

    # Instructions
    canvas.create_text(335, 190, text="Input one ticker for DD. Input two tickers for a comparision", anchor="center", fill="white", font=("Helvetica", 15, "bold"))

    # Execute button
    submitText = tk.StringVar()
    submitBtn = tk.Button(root, textvariable=submitText, command=execute, font = ("Helvetica", 13, "bold"))
    submitText.set("Execute Query")
    submitBtnPlace = canvas.create_window(335, 260, anchor="center", window=submitBtn)

    # Excel output checkbox
    excelFlag = tk.BooleanVar()
    cb = tk.Checkbutton(root, text="Dashboard Output", variable=excelFlag, onvalue=True, offvalue=False, font = ("Helvetica", 13, "bold"))
    cbPlace = canvas.create_window(335, 330, anchor="center", window=cb)    

    # Quit button
    quitBtn = tk.Button(root, text="Exit", command=clickExitButton, font = ("Helvetica", 13, "bold"))
    quitBtnPlace = canvas.create_window(335, 400, anchor="center", window=quitBtn)

    # Ticker Boxes
    ticker1Entry = tk.Entry(root, font=("Helvetica", 12), width=6)
    ticker2Entry = tk.Entry(root, font=("Helvetica", 12), width=6)
    ticker1Window = canvas.create_window(112, 300, anchor="center", window=ticker1Entry)
    ticker1Window = canvas.create_window(558, 300, anchor="center", window=ticker2Entry)
    ticker1Entry.insert(0, "Ticker 1")
    ticker2Entry.insert(0, "Ticker 2")

    # Label for entry boxes
    canvas.create_text(112, 260, text="First Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))
    canvas.create_text(558, 260, text="Second Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))

    # TODO
    """
    Change Execute to working when in progress
    Make execute button actually do something
    Make it look good

    """

    # Bind entry boxes
    ticker1Entry.bind("<Button-1>", entry_clear)
    ticker2Entry.bind("<Button-1>", entry_clear)

    # End GUI code
    root.mainloop()

    #return ticker1, ticker2, excelFlag


def main():

    # Get ticker(s) and check for excel output flag
    """
    # Initiate class(es) for storing stock data
    ticker1 = stockData(guiReturn[0])
    
    # Only initiate ticker 2 if user supplies value
    if guiReturn[1] != "":
        ticker2 = stockData(guiReturn[1])
    else:
        ticker2 = ""
    excelFlag = guiReturn[2]
    """


cikList = get_CIK_list()
guiReturn = get_tickers()
#if __name__ == "__main__":
    #main()