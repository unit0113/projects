import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
import requests
import urllib
import edgarquery
import time
        

def get_CIK_list():
    """ Webscrape list of stock tickers and their CIK numbers, ouput into a dictionary

    Returns:
        cikList (dict): dictionary of pairs of stock tickers and CIK numbers
    """

    # Initial CIK data pull
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            cikData = urllib.request.urlopen("https://www.sec.gov/include/ticker.txt")    
        except:
            time.sleep(0.1)
            continue
        else:
            break
    else:
        tk.messagebox.showerror(title="Error", message="Network Error. Please try again")
    
    # Parse into dict
    cikList = {}
    for line in cikData:
        line = line.decode("utf-8")
        (key, val) = line.split()
        cikList[key] = val

    return cikList    


def get_tickers(cikList):
    """ Function to open up a GUI that allows the user to select options. Can input one ticker for single stock DD.
    Can input two tickers for comparision. Can select a flag that will create a CSV for inport into excel dashboard. Also a quit option to abort execution of program

    Returns:
        guiReturn (lst): first symbol, first CIK, second symbol, second CIK, excel export flag
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
        if ticker1Entry.get() == "Ticker 1":
            ticker1Entry.delete(0, tk.END)
        if ticker2Entry.get() == "Ticker 2":
            ticker2Entry.delete(0, tk.END)
    

    # Update autocomplete list box
    def update(data, src):
        
        # Clear list box
        src.delete(0,tk.END)
    
        # Add tickers to list
        for ticker in data:
            src.insert(tk.END, ticker.upper())
    

    # Fill entry box from selection in list box
    def fillout(e, src, src2):
        
        # Delete current entry
        src.delete(0,tk.END)
        #ticker1Entry.delete(0,tk.END)

        # Select from list box
        src.insert(0, src2.get(tk.ACTIVE))
        #ticker1Entry.insert(0, ticker1List.get(tk.ACTIVE))
    

    # Autocomplete
    def autocomplete(e, src, src2):
        
        # Grab typed data
        typed= src2.get()
        
        # Check if anything has been entered
        if typed == '':
            data = tickerList
        
        # Update list
        else:
            data=[]
            for item in tickerList:
                if typed.lower() in item.lower()[0:len(typed)]:
                    data.append(item.upper())
        update(data, src)    
    

    # Pull ticker and execute rest of program
    def execute():
        
        # Pull tickers and initial ticker check
        if not ticker1Entry.get() or ticker1Entry.get() == "Ticker 1":
            ticker1 = ""
            cik1 = ""
        else:
            try:
                ticker1 = ticker1Entry.get().lower()
                cik1 = int(cikList[ticker1])
            except:
                tk.messagebox.showwarning(title="Invalid Ticker", message="Ticker 1 is invalid, please enter a valid ticker")
                return
        if not ticker2Entry.get() or ticker2Entry.get() == "Ticker 2":
            ticker2 = ""
            cik2 = ""
        else:
            try:
                ticker2 = ticker2Entry.get().lower()
                cik2 = int(cikList[ticker2])
            except:
                tk.messagebox.showwarning(title="Invalid Ticker", message="Ticker 2 is invalid, please enter a valid ticker")
                return

        # If no tickers
        if ticker1 == "" and ticker2 == "":
            tk.messagebox.showwarning(title="No Tickers Provided", message="Please Provide at least one Stock ticker")
            return                  
        
        #Change text on execute button
        submitText.set("Working...")

        # Assembles values and pass to edgarquery
        guiReturn = [ticker1, cik1, ticker2, cik2, excelFlag.get()]
        edgarquery.main(guiReturn)
        
        # Offer to run new query or exit
        response = tk.messagebox.askokcancel(title="Complete", message="Query Complete, select OK for new query or press cancel to exit")
        if response == True:
            submitText.set("Execute Query")
        else:
            sys.exit()


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

    # List Boxes
    ticker1List = tk.Listbox(root, width=8)
    ticker1ListWindow = canvas.create_window(112, 405, anchor="center", window=ticker1List)
    ticker2List = tk.Listbox(root, width=8)
    ticker2ListWindow = canvas.create_window(558, 405, anchor="center", window=ticker2List)
    tickerList = [*cikList]
    update(tickerList, ticker1List)
    update(tickerList, ticker2List)

    # Label for entry boxes
    canvas.create_text(112, 260, text="First Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))
    canvas.create_text(558, 260, text="Second Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))

    # Bind entry boxes
    ticker1Entry.bind("<Button-1>", entry_clear)
    ticker2Entry.bind("<Button-1>", entry_clear)

    # Bind for clicking on list items
    ticker1List.bind("<<ListboxSelect>>", lambda event, arg=ticker1Entry, arg2=ticker1List: fillout(event, arg, arg2))
    ticker2List.bind("<<ListboxSelect>>", lambda event, arg=ticker2Entry, arg2=ticker2List: fillout(event, arg, arg2))
    
    # Bind for autocomplete
    ticker1Entry.bind("<KeyRelease>", lambda event, arg=ticker1List, arg2=ticker1Entry: autocomplete(event, arg, arg2))
    ticker2Entry.bind("<KeyRelease>", lambda event, arg=ticker2List, arg2=ticker2Entry: autocomplete(event, arg, arg2))
    
    # End GUI code
    root.mainloop()


def main():
    """ Runs two subprograms. Webscrapes ticker/CIK paires from SEC site, gets tickers from user and passes the CIK codes and a seperate flag to the edgarquery program

    Returns:
    guiReturn (lst): first symbol, first CIK, second symbol, second CIK, excel export flag
    """
    # Run subprograms
    cikList = get_CIK_list()
    get_tickers(cikList)


if __name__ == "__main__":
    main()


# TODO
"""
Make an executable using PyInstaller
remove error checker for running edgarquery
add network error checker

for _ in range(MAX_RETRIES):
    try:
        # ... do stuff ...
    except SomeTransientError:
        time.sleep(1)
        continue
    else:
        break
else:
    raise PermanentError()


"""