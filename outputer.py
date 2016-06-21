from __future__ import print_function
import datetime
import os
import sys

global tee_file
tee_file = None

# http://stackoverflow.com/questions/29772158/make-ipython-notebook-print-in-real-time
class flushfile():
    def __init__(self, f):
        self.f = f
    def __getattr__(self,name): 
        return object.__getattribute__(self.f, name)
    def write(self, x):
        global tee_file
        if tee_file:
            tee_file.write(x)
        self.f.write(x)
        self.f.flush()
    def flush(self):
        self.f.flush()

setup_flush = True
try:
    if oldsystdout:
        print("Already setup flushfile/tee")
        setup_flush = False
except NameError:
    setup_flush = True

if setup_flush:
    oldsysstdout = sys.stdout
    sys.stdout = flushfile(oldsysstdout)

class TeeOutput():
    def __init__(self, path):
        self.path = path
    def __enter__(self):
        self.fd = open(self.path, "w")
        global tee_file
        tee_file = self.fd
        return self.fd
    def __exit__(self, type, value, traceback):
        global tee_file
        tee_file.flush()
        self.fd.close()
        if tee_file == self.fd:
            tee_file = None

def setup_directory(*path_parts):
    path = os.path.join(*path_parts)
    try:
        os.makedirs(path)
    except OSError as e:
        pass
    return path

def timestamp(base_name=None, extension=None):
    name = datetime.datetime.now().strftime("%Y-%m-%d~%H_%M_%S_%f")[0:-3]
    if base_name:
        name = base_name + name
    if extension:
        name = name + "." + extension
    return name
