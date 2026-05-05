#config.py 
# Configuration file for chiplet disaggregation project

sram_selection_mode = "random" # or "max"

calibration_mode = "min_median" # "avg" or "max_minus_min" or "max" or "mean_std" or "min_median"


#Prints 
#main.py
print_info = False #Set to True to print debug info
fast_test = False #Impact only CycleAccurate Sim - True - skip running cycle accurate simulator use value = W*H^2 , False - Will run cycle accurate simulator ; Energy is accurate here
latency_en = True #Impact both CycleAccurate Sim and Energy : True - Will obtain cycle accurate simulator numbers, False - Will use 0 for latency and energy from simulator
