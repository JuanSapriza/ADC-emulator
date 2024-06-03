



import itertools
from timeseries import *

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

    def populate(self):
        keys                    = self.params_list_dict.keys()
        values                  = self.params_list_dict.values()
        self.params_list        = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        self.outputs_count      = len(self.params_list)
        print(f"Populated {self.name} and would generate {self.outputs_count} outputs")

    def run(self, count=0):
        for in_signal in self.inputs:
            for params in self.params_list:
                self.outputs.append( self.operation( in_signal, params ) )
                self.outputs[-1].params[ TS_PARAMS_STEP_HISTORY ].append(self.name)
                self.outputs[-1].params[ TS_PARAMS_INPUT_SERIES ]   = in_signal
                self.outputs[-1].params[ TS_PARAMS_OPERATION ]      = self.operation
                print(f"\r{count}", end=" ")
                count += 1
        return count

    def init(self, inputs, count ):
        self.inputs = inputs
        self.run( count )
        return count




def run_steps_recursive( parent, count=0 ):
    for child in parent.children_steps:
        child.inputs = parent.outputs
        count = child.run(count)
        if child.children_steps != []: count = run_steps_recursive( child, count )
    return count


def get_last_generation_recursive( parent, last_gen_list ):
    if parent.children_steps == []:
        last_gen_list.append(parent)
    else:
        for child in parent.children_steps:
            get_last_generation_recursive( child, last_gen_list )

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


