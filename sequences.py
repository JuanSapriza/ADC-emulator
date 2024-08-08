import tracemalloc
import itertools
import time
import signal
from timeseries import *
from copy import deepcopy
from logger import *

TIMEOUT_S = 5

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

# Function to set the timeout
def set_timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

def get_id():
    i = 0
    while(True):
        yield i
        i  += 1

class Step()    :
    def __init__(self, name, operation, params_list_dict ):
        self.name                   = name
        self.operation              = operation
        self.params_list_dict       = params_list_dict
        self.params_list            = []
        self.children_steps         = []
        self.children_steps_left    = 0
        self.inputs                 = []
        self.outputs                = []
        self.outputs_count          = 0
        self.latency                = 0
        self.count                  = 0

    def populate(self):
        keys                        = self.params_list_dict.keys()
        values                      = self.params_list_dict.values()
        self.params_list            = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        self.outputs_count          = len(self.params_list)
        self.children_steps_left    = len(self.children_steps)
        log(f"Populated {self.name} and would generate {self.outputs_count} outputs")

    def run(self, count=0, kamikaze=False):
        for in_signal in self.inputs:
            for params in self.params_list:

                try:
                    set_timeout(TIMEOUT_S)
                    start_time = time.time()  # Capture start time
                    output = self.operation(in_signal, params)
                    end_time = time.time()    # Capture end time
                    latency = end_time - start_time
                except TimeoutError:
                    log(f"❗⏳❗ {self.name} for {in_signal} and paramaters {params} took longer than {TIMEOUT_S} s!! {len(in_signal.time)} | {len(in_signal.data)} | Skipped :/")
                    in_signal.print_params()
                finally:
                    signal.alarm(0)


                self.latency += latency

                if output != None:
                    output.params[ TS_PARAMS_STEP_HISTORY ]     .append(self.name)
                    output.params[ TS_PARAMS_LATENCY_HISTORY ]  .append(latency)
                    output.params[ TS_PARAMS_OPERATION ]        = self.operation
                    output.params[ TS_PARAMS_INPUT_SERIES ]     = in_signal.params[TS_PARAMS_ID]
                    output.generate_unique_id()
                    if kamikaze and self.children_steps_left == 0:
                        output.data = []
                        output.time = []

                    self.outputs.append( output )

                count += 1
                self.count += 1
                log(f"\r{self.name}: {count}", end=" ")
        log(f"\n✅\t{self.name}\tOutput {len(self.outputs)} timeseries.\tTook {self.latency:0.3f} s",f"({self.latency/len(self.outputs):0.3f} s/Ts)." if len(self.outputs) != 0 else "")
        update_catalog(self.outputs)
        return count

    def copy(self):
        return deepcopy(self)


    def init(self, inputs, count ):
        self.inputs = inputs
        count = self.run( count )
        return count

def run_steps( step, initial_signals ):
    def run_steps_recursive( parent, count=0 ):
        for child in parent.children_steps:
            child.inputs = parent.outputs
            count = child.run(count)
            if child.children_steps != []: count = run_steps_recursive( child, count )
        return count

    count = 0
    count = step.init( initial_signals, count )
    run_steps_recursive( step, count )
    log("\rDone!\n")

def get_last_outputs( step ):
    def get_last_generation_recursive( parent, last_gen_list ):
        if parent.children_steps == []:
            last_gen_list.extend(parent.outputs)
        else:
            for child in parent.children_steps:
                get_last_generation_recursive( child, last_gen_list )
    last_gen = []
    get_last_generation_recursive( step, last_gen )
    return last_gen

def populate_recursive( parent ):
    parent.populate()
    for child in parent.children_steps:
        populate_recursive( child )

def get_run_length_recursive( parent ):
    length = 1
    for child in parent.children_steps:
        length += get_run_length_recursive( child )
    return length*parent.outputs_count

def get_output_count_recursive( parent ):
    length = 1
    if parent.children_steps != []:
        length = 0
        for child in parent.children_steps:
            length += get_output_count_recursive( child )
    return length*parent.outputs_count

def operation_buffer( series, params ):
    return series

def filter_timeseries(timeseries_list, params_dict):
    def matches_params(timeseries, params_dict):
        for key, value in params_dict.items():
            if isinstance(value, list):
                # Check if any of the values in the list match the timeseries params
                if key not in timeseries.params or not any(item in timeseries.params[key] for item in value):
                    return False
            else:
                # Check if the single value matches the timeseries params
                if key not in timeseries.params or timeseries.params[key] != value:
                    return False
        return True

    filtered_timeseries = [ts for ts in timeseries_list if matches_params(ts, params_dict)]
    return filtered_timeseries

# def get_child_from_step(series, step):
#     """
#     Get a previous series from the current serie's history.

#     :param series: The grand-child series
#     :param step: The step whose output you are looking for
#     :return: The series ID from the input one's history whose last step was the one specified. If
#     the desired step is not found, None is returned.
#     """
#     while( True ):
#         if series.params[TS_PARAMS_STEP_HISTORY][-1] == step:
#             break
#         else:
#             series = series.params[TS_PARAMS_INPUT_SERIES]
#         if len(series.params[TS_PARAMS_STEP_HISTORY]) == 1:
#             return None

#     return series



def get_all_steps_recursive(parent_step):
    """
    Recursively collect all steps starting from the parent step.

    :param parent_step: The root step from which to start collecting.
    :return: A list of all steps.
    """
    all_steps = []

    def collect_steps(step):
        all_steps.append(step)
        for child in step.children_steps:
            collect_steps(child)

    collect_steps(parent_step)
    return all_steps


def run_and_save( initial_step, input_signals, filename, kamikaze=False ):
    populate_recursive( initial_step )
    log(f"Will input {len(input_signals)} series, do a max. of {get_run_length_recursive( initial_step )*len(input_signals)} runs, generating a total of {get_output_count_recursive( initial_step )*len(input_signals)} output signals")

    run_steps( initial_step, input_signals )

    last_outputs = get_last_outputs( initial_step)

    if kamikaze:
        for r in last_outputs:
            r.limit_to_metadata()

    steps = get_all_steps_recursive(initial_step)

    sum = 0
    for s in steps:
        log(f"{s.latency:0.3f} s\t{s.name}")
        sum += s.latency
    log(f"----------------\n    {sum:0.3f} s for {len(last_outputs)} outputs\n({sum/len(last_outputs):0.3f} s/output)")

    save_series(last_outputs, filename, input_series = input_signals )