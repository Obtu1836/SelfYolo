import torch as th

def get_device(device: str):
    if device == 'cpu':
        return 'cpu'
    if device == 'cuda':
        if th.cuda.is_available():
            return 'cuda'
        if th.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if th.cuda.is_available():
        return 'cuda'
    if th.backends.mps.is_available():
        return 'mps'
    return 'cpu'
