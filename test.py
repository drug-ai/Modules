from operator import *
from modules.utils_fn import try_all_gpus

device = try_all_gpus()
print(device)

