# type: ignore
import os
from pathlib import Path
import pickle

def get_object(name):
    objects_dir = os.path.join(Path(__file__).parent.absolute(), 'objects')
    if not name.endswith('.pkl'):
        name = name+".pkl"
    filepath = os.path.join(objects_dir, name)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj
