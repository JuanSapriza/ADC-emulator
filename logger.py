import os
import sys
import datetime


log_directory   = ''
filename        = ''
filepath        = ''
log_buffer      = ''


def new_log():
    global filename
    global filepath
    global log_buffer
    log_directory = './log'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    filepath = os.path.join(log_directory, filename)
    log_buffer = []

def log(message, end="\n"):
    global log_buffer
    # Adds to buffer and optionally print to console
    if "\r" not in end: log_buffer.append(message + end)
    print(message, end=end)

def dump_log():
    global log_buffer

    log(f"\n\n--------------------------\n{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n--------------------------\n\n")

    # Write buffer to file and clear it
    with open(filepath, 'a') as file:
        file.writelines(log_buffer)
    log_buffer = []

def get_log_name():
    global filename
    return filename