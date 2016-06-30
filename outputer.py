from __future__ import print_function

import datetime
import ipywidgets
import os
import sys
import traceback

from IPython.display import display

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

class TeeOutput(object):
    """Wrapper to pipe standard output into a file as well as the console using 'with'."""
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
    """Create the directory if it doesn't exist yet."""
    path = os.path.join(*path_parts)
    try:
        os.makedirs(path)
    except OSError as e:
        pass
    return path

def timestamp(base_name=None, extension=None):
    """Get a canonical timestamp suitable for file names."""
    name = datetime.datetime.now().strftime("%Y-%m-%d~%H_%M_%S_%f")[0:-3]
    if base_name:
        name = base_name + name
    if extension:
        name = name + "." + extension
    return name

def show_progress(description, steps):
    progress_bar = ipywidgets.FloatProgress(
        min=0, max=steps, description=description
    )
    display(progress_bar)
    return progress_bar

class ProgressTracker(object):
    def __init__(self, titles, steps, output_path, average_window=100):
        self.titles = titles
        self.steps = steps
        self.average_window = average_window
        self.results = []
        self.output_path = setup_directory(output_path)

        self.setup_widgets()

    def setup_label(self, title):
        return ipywidgets.FloatText(value=0, description=title, disabled=True)

    def setup_widgets(self):
        """Set up widgets used to display current/average progress."""

        # Set up to show a progress bar so you some mesure of time required.
        self.progress_bar = show_progress("Graph Steps:", self.steps)

        # Set up to show current training results as well as a running average.
        self.current_display = [self.setup_label(title) for title in titles]
        self.average_display = [self.setup_label(" ") for _ in titles]
        current_title_html = "<div style=""margin-left:90px"">Current</div>"
        average_title_html = "<div style=""margin-left:90px"">Running Average</div>"
        display(ipywidgets.HBox([
            ipywidgets.Box([ipywidgets.HTML(current_title_html)] + self.current_display),
            ipywidgets.Box([ipywidgets.HTML(average_title_html)] + self.average_display)
        ]))

    def setup_eval(self, stack_data):
        self.timestamp = timestamp()
        self.stack_data = stack_data

    def start_eval(self, stack, graph_info):
        with open(os.path.join(self.output_path, self.timestamp + ".xml"), "w") as text_file:
            text_file.write(stack_data)
        stack.checkpoint_path(os.path.join(self.output_path, self.timestamp + ".ckpt"), graph_info)
        self.results = []

    def record_score(self, score):
        self.results.append(score)
        self.display_score(score)

    def display_score(self, score):
        for display, value in zip(self.current_display, score):
            if value is not None:
                display.value = value

        resultCount = min(len(results), 100)
        last_results = zip(*results[-resultCount:])
        filtered = [[v for v in x if v is not None] for x in last_results]
        for display, values in zip(self.average_display, filtered):
            count = len(values)
            if count:
                display.value = sum(values) / count

    def output(self):
        with open(os.path.join(self.output_path, self.timestamp + ".csv"), "w") as text_file:
            text_file.write(",".join(self.titles) + "\n")
            for score in self.results:
                text_file.write((",".join(str(v) for v in score)) + "\n")
        print("Saved results:", timestamp)
        self.results = []

    def error(self, exc_info):
        exc_type, exc_value, exc_traceback = exc_info
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print(lines[-1])

        error_file = "ERR~" + self.timestamp + ".txt"
        with open(os.path.join(self.output_path, error_file), "w") as text_file:
            text_file.write(self.stack_data)
            text_file.write("\n------------------------------------------------------------\n")
            for line in error_data:
                text_file.write(line)

    def update_progress(self, value):
        self.progress_bar.value = value
