import tkinter as tk
import tkinter.filedialog
from PIL import ImageTk
import sys
import urllib
import edgarquery
import json
from shutil import copyfile


global headers
headers = {}

@edgarquery.network_check_decorator(1)
def get_CIK_list():
    """ Webscrape list of stock tickers and their CIK numbers, ouputs into a dictionary

    Returns:
        cikList (dict): dictionary of pairs of stock tickers and CIK numbers
    """
    cikData = urllib.request.urlopen("https://www.sec.gov/include/ticker.txt") 
    # Parse into dict
    cik_list = {}
    for line in cikData:
        line = line.decode("utf-8")
        (key, val) = line.split()
        cik_list[key] = val

    return cik_list   


def get_tickers(cikList):
    """ Function to open up a GUI that allows the user to select options. Can input one ticker for single stock DD.
    Can input two tickers for comparision. Can select a flag that will create a CSV for inport into excel dashboard. Functionality for creating a User Agent. 
    Also a quit option to abort execution of program

    Returns:
        guiReturn (lst): first symbol, first CIK, second symbol, second CIK, excel export flag
        headers (dict): Parameters for User Agent
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

    # Imports User-Agent to program to allow access to EDGAR. Returns User-Agent as specified at https://www.sec.gov/os/webmaster-faq
    def import_user_agent():
        global headers
        canvas.delete("agent_tag")
        # Open file
        try:
            with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt') as f:
                data = f.read()
                headers = json.loads(data)
                canvas.create_text(335, 480, text='User Agent found', anchor="center", justify='center', fill="green2", tag="agent_tag", font=("Helvetica", 12, "bold"))
        except:
            canvas.create_text(335, 480, text="No User Agent found.\nPlease add or create a User Agent", anchor="center", justify='center', fill="red", tag="agent_tag", font=("Helvetica", 12, "bold"))
            return

        # Check for Valid Agent
        if headers["Accept-Encoding"] != "gzip, deflate" or headers["Host"] != "www.sec.gov":
            canvas.delete("agent_tag")
            canvas.create_text(335, 480, text="Invalid User agent", anchor="center", justify='center', fill="red", tag="agent_tag", font=("Helvetica", 12, "bold"))


    # Function for exiting GUI
    def clickExitButton():
        sys.exit()


    # Delete contents of entry boxes when clicked on
    def entry_clear(e):
        if ticker1_entry.get() == "Ticker 1":
            ticker1_entry.delete(0, tk.END)
        if ticker2_entry.get() == "Ticker 2":
            ticker2_entry.delete(0, tk.END)
    

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
            data = ticker_list
        
        # Update list
        else:
            data=[]
            for item in ticker_list:
                if typed.lower() in item.lower()[0:len(typed)]:
                    data.append(item.upper())
        update(data, src)    
    

    # Browse for User Agent
    def browse():
        file = (tkinter.filedialog.askopenfilename())
        copyfile(file, r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt')
        import_user_agent()


    # Create new User Agent
    def create():
        
        def create_new_agent():
            # Pull data
            name = name_entry.get()
            email = email_entry.get()
            agent = {"User-Agent": "For Personal Use of " + name + " " + email,
                     "Accept-Encoding": "gzip, deflate",
                     "Host": "www.sec.gov"}
            
            # Save User Agent
            with open(r'C:\Users\unit0\OneDrive\Desktop\EDGAR\user_agent.txt', 'w') as f:
                print(json.dumps(agent), file=f)

            import_user_agent()
            create_root.destroy()

        
        # Delete contents of entry boxes when clicked on
        def create_entry_clear(e):
            if name_entry.get() == "First Last":
                name_entry.delete(0, tk.END)
            if email_entry.get() == "Email Address":
                email_entry.delete(0, tk.END)
        

        create_root = tk.Tk()
        create_root.title("Create User Agent")

        # Size of GUI
        gui_width = 250
        gui_height = 200

        # Size of screen
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()

        # Place GUI in middle of screen
        x = (screenWidth / 2) - (gui_width / 2)
        y = (screenHeight / 2) - (gui_height / 2)
        create_root.geometry(f'{gui_width}x{gui_height}+{int(x)}+{int(y)}')

        #Create Window
        create_canvas = tk.Canvas(create_root, width=gui_width, height=gui_height)
        create_canvas.pack(fill="both", expand="True")
        create_root.resizable(width=False, height=False)

        # Create Button
        create_btn_2 = tk.Button(create_root, text='Create', command=create_new_agent, font = ("Helvetica", 13, "bold"))
        create_canvas.create_window(125, 175, anchor="center", window=create_btn_2)

        # Field Labels
        create_canvas.create_text(25, 25, text="Name (First Last)", anchor="w", fill="black", font=("Helvetica", 13, "bold"))
        create_canvas.create_text(25, 100, text="Email Address", anchor="w", fill="black", font=("Helvetica", 13, "bold"))

        # Input Boxes
        name_entry = tk.Entry(create_root, font=("Helvetica", 12), width=25)
        email_entry = tk.Entry(create_root, font=("Helvetica", 12), width=25)
        create_canvas.create_window(125, 50, anchor="center", window=name_entry)
        create_canvas.create_window(125, 125, anchor="center", window=email_entry)
        name_entry.insert(0, "First Last")
        email_entry.insert(0, "Email Address")

        # Bind entry boxes
        name_entry.bind("<Button-1>", create_entry_clear)
        email_entry.bind("<Button-1>", create_entry_clear)
    
    
    # Pull ticker and execute rest of program
    def execute():
        global headers
        stocks = []

        # Pull tickers and initial ticker check
        if not ticker1_entry.get() or ticker1_entry.get() == "Ticker 1":
            ticker1 = ""
            cik1 = ""
        else:
            try:
                ticker1 = ticker1_entry.get().lower()
                cik1 = int(cikList[ticker1])
            except:
                tk.messagebox.showwarning(title="Invalid Ticker", message="Ticker 1 is invalid, please enter a valid ticker")
                return
        if ticker1 != "":
            stock1 = [ticker1, cik1]
            stocks.append(stock1)

        if not ticker2_entry.get() or ticker2_entry.get() == "Ticker 2":
            ticker2 = ""
            cik2 = ""
        else:
            try:
                ticker2 = ticker2_entry.get().lower()
                cik2 = int(cikList[ticker2])
            except:
                tk.messagebox.showwarning(title="Invalid Ticker", message="Ticker 2 is invalid, please enter a valid ticker")
                return
        if ticker2 != "":
            stock2 = [ticker2, cik2]
            stocks.append(stock2)

        # If no tickers
        if ticker1 == "" and ticker2 == "":
            tk.messagebox.showwarning(title="No Tickers Provided", message="Please Provide at least one Stock ticker")
            return                  
        
        #Change text on execute button
        submit_text.set("Working...")

        # Assembles values and pass to edgarquery
        gui_return = [stocks, excel_flag.get()]
        edgarquery.main(gui_return, headers)
        
        # Offer to run new query or exit
        response = tk.messagebox.askokcancel(title="Complete", message="Query Complete, select OK for new query or press cancel to exit")
        if response == True:
            submit_text.set("Execute Query")
        else:
            sys.exit()


    # Set GUI size and layout
    canvas = tk.Canvas(root, width=gui_width, height=gui_height)
    canvas.pack(fill="both", expand="True")
    root.resizable(width=False, height=False)

    # Background
    bg = ImageTk.PhotoImage(file=".venv/programs/EDGAR/bg.png")
    canvas.create_image(0,0, image=bg, anchor="nw")

    # Pull User Agent
    import_user_agent()

    # Instructions
    canvas.create_text(335, 190, text="Input one ticker for DD. Input two tickers for a comparision", anchor="center", fill="white", font=("Helvetica", 15, "bold"))

    # Execute button
    submit_text = tk.StringVar()
    submit_btn = tk.Button(root, textvariable=submit_text, command=execute, font = ("Helvetica", 13, "bold"))
    submit_text.set("Execute Query")
    canvas.create_window(335, 260, anchor="center", window=submit_btn)

    # Excel output checkbox
    excel_flag = tk.BooleanVar()
    cb = tk.Checkbutton(root, text="Dashboard Output", variable=excel_flag, onvalue=True, offvalue=False, font = ("Helvetica", 13, "bold"))
    canvas.create_window(335, 330, anchor="center", window=cb)    

    # Quit button
    quit_btn = tk.Button(root, text="Exit", command=clickExitButton, font = ("Helvetica", 13, "bold"))
    canvas.create_window(335, 400, anchor="center", window=quit_btn)

    # Ticker Boxes
    ticker1_entry = tk.Entry(root, font=("Helvetica", 12), width=6)
    ticker2_entry = tk.Entry(root, font=("Helvetica", 12), width=6)
    canvas.create_window(112, 300, anchor="center", window=ticker1_entry)
    canvas.create_window(558, 300, anchor="center", window=ticker2_entry)
    ticker1_entry.insert(0, "Ticker 1")
    ticker2_entry.insert(0, "Ticker 2")

    # List Boxes
    ticker1List = tk.Listbox(root, width=8)
    canvas.create_window(112, 405, anchor="center", window=ticker1List)
    ticker2List = tk.Listbox(root, width=8)
    canvas.create_window(558, 405, anchor="center", window=ticker2List)
    ticker_list = [*cikList]
    update(ticker_list, ticker1List)
    update(ticker_list, ticker2List)

    # Label for entry boxes
    canvas.create_text(112, 260, text="First Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))
    canvas.create_text(558, 260, text="Second Ticker", anchor="center", fill="white", font=("Helvetica", 13, "bold"))

    # Browse button for User Agent
    browse_text = tk.StringVar()
    browse_btn = tk.Button(root, textvariable=browse_text, command=browse, font = ("Helvetica", 11, "bold"))
    browse_text.set("Browse")
    canvas.create_window(280, 520, anchor="center", window=browse_btn)

    # Create button for User Agent
    create_text = tk.StringVar()
    create_btn = tk.Button(root, textvariable=create_text, command=create, font = ("Helvetica", 11, "bold"))
    create_text.set("Create")
    canvas.create_window(390, 520, anchor="center", window=create_btn)

    # Bind entry boxes
    ticker1_entry.bind("<Button-1>", entry_clear)
    ticker2_entry.bind("<Button-1>", entry_clear)

    # Bind for clicking on list items
    ticker1List.bind("<<ListboxSelect>>", lambda event, arg=ticker1_entry, arg2=ticker1List: fillout(event, arg, arg2))
    ticker2List.bind("<<ListboxSelect>>", lambda event, arg=ticker2_entry, arg2=ticker2List: fillout(event, arg, arg2))
    
    # Bind for autocomplete
    ticker1_entry.bind("<KeyRelease>", lambda event, arg=ticker1List, arg2=ticker1_entry: autocomplete(event, arg, arg2))
    ticker2_entry.bind("<KeyRelease>", lambda event, arg=ticker2List, arg2=ticker2_entry: autocomplete(event, arg, arg2))
    
    # End GUI code
    root.mainloop()


def main():
    """ Runs two subprograms. Webscrapes ticker/CIK paires from SEC site,
        and gets tickers from user and passes the CIK codes and a seperate flag to the edgarquery program

    Returns:
    gui_return (lst): first symbol, first CIK, second symbol, second CIK, excel export flag
    headers (dict): Parameters for User Agent
    """
    # Run subprograms
    cikList = get_CIK_list()
    get_tickers(cikList)


if __name__ == "__main__":
    main()


"""
TODO
Make an executable using PyInstaller
make user agent file paths programitically
figure out how to make work for ADR's
email address checker
"""