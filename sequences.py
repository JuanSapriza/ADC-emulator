import itertools


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

    def run(self):
        for in_signal in self.inputs:
            for params in self.params_list:
                self.outputs.append( self.operation( in_signal, params ) )

    def init(self, inputs ):
        self.inputs = inputs
        self.run()




def run_steps_recursive( parent ):
    for child in parent.children_steps:
        child.inputs = parent.outputs
        child.run()
        if child.children_steps != []: run_steps_recursive( child )


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


