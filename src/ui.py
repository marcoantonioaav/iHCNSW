import os

def print_indexing_frame(index, size):
    clear_terminal()
    print("  Indexing  ")
    print(get_loading_bar_string(index, size))
    print(f"Done: {index}/{size}")

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
   
def get_number_of_bars(index, size):
    return min(int((10*(index+1))/size), 10)

def get_loading_bar_string(index, size):
    bar = ""
    bar_number = get_number_of_bars(index, size)
    for i in range(bar_number):
        bar += "▪"
    for i in range(10 - bar_number):
        bar += "▫"
    return f"[{bar}]"    

