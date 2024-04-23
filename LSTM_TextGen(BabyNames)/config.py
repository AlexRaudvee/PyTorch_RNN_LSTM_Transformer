import torch 
import unidecode

# set device
"""
set 'cpu' if you are using not MacOS M1 or M2 chip
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

"""
here you have to input the path to the txt file
"""
path_to_file = ''

file = unidecode.unidecode(open(f"{path_to_file}").read())