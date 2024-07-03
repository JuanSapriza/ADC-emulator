


import tracemalloc
import itertools
import time
from timeseries import *
from copy import deepcopy

class Step()    :
    def __init__(self, name, operation, params_list_dict ):
        self.name               = name
        self.operation          = operation
        self.params_list_dict   = params_list_dict
        self.params_list        = []
        self.children_steps     = []
        self.inputs             = []
        self.outputs            = []
        self.outputs_count      = 0
        self.latency            = 0
        self.count              = 0

    def populate(self):
        keys                    = self.params_list_dict.keys()
        values                  = self.params_list_dict.values()
        self.params_list        = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        self.outputs_count      = len(self.params_list)
        print(f"Populated {self.name} and would generate {self.outputs_count} outputs")

    def run(self, count=0):
        for in_signal in self.inputs:
            for params in self.params_list:
                start_time = time.time()  # Capture start time
                output = self.operation(in_signal, params)
                end_time = time.time()    # Capture end time
                latency = end_time - start_time

                self.latency += latency

                if output != None:
                    output.params[ TS_PARAMS_STEP_HISTORY ].append(self.name)
                    output.params[ TS_PARAMS_LATENCY_HISTORY ].append(latency)
                    output.params[ TS_PARAMS_INPUT_SERIES ]     = in_signal
                    output.params[ TS_PARAMS_OPERATION ]        = self.operation
                    self.outputs.append( output )

                count += 1
                self.count += 1
                print(f"\r{count}", end=" ")
                # print(f"{self.name} \t {count}\t({self.mycounts})")
        print(f"\n✅\t{self.name}\tOutput {len(self.outputs)} timeseries.\tTook {self.latency:0.3f} s",f"({self.latency/len(self.outputs):0.3f} s/Ts)." if len(self.outputs) != 0 else "")
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
    print("\rDone!\n")

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
