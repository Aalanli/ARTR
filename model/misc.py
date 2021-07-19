import math

def calculate_param_size(model):
    params = 0
    for i in model.parameters():
        params += math.prod(list(i.shape))
    return params