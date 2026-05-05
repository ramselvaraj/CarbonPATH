import numpy as np
from   tqdm import tqdm, trange
import pandas as pd 
import itertools as it 
from   matplotlib import pyplot as plt

#Importing all functions from CO2_func files
from .CO2_func import *
from .tech_scaling import *
from config import print_info 

import argparse
import json
import ast
import os

debug = False

def find_carbon(args):
    
    scaling_factors = load_tables()


   
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    design_dir = os.path.join(base_dir, "arch_params/")
    
    print(f"Using design_dir: {design_dir}") if print_info else None 
    
    architecture_file = design_dir+'architecture.json'
    designC_file = design_dir+'designC.json'
    operationalC_file = design_dir+'operationalC.json'
    packageC_file = design_dir+'packageC.json'
    print(" ---------------------------------------------------------") if print_info else None
    print("Using below files for CFP estimations : \n") if print_info else None
    print(architecture_file) if print_info else None
    print(designC_file) if print_info else None
    print(operationalC_file) if print_info else None
    print(packageC_file) if print_info else None
    print(" ---------------------------------------------------------") if print_info else None


    with open(architecture_file,'r') as json_file:
        config_json = json.load(json_file)

    

    if args.design is not None:
        design = args.design
        print("Design provided as DataFrame.") if print_info else None
    elif args.design_file is not None:
        design = pd.DataFrame(config_json).T
    print(f"type of design before pkg_type drop: {type(design)}") if print_info else None
    print(design) if print_info else None 
    
    if args.pkg_type is not None:
        package_type = args.pkg_type
    else:
        package_type = design.loc['pkg_type']
    print(f"Package Type : {package_type}") if print_info else None
    
    print(f"type of design after pkg_type drop: {type(design)}") if print_info else None
    print(design) if print_info else None

    with open(designC_file, 'r') as f:
        designC_values = json.load(f)
    num_iter = designC_values['num_iter']
    num_prt_mfg = designC_values['num_prt_mfg']
    transistors_per_gate = designC_values['Transistors_per_gate']
    power_per_core = designC_values['Power_per_core']
    carbon_per_kWh = designC_values['Carbon_per_kWh']

    print(" ") if print_info else None

    if args.lifetime is not None:
        lifetime = args.lifetime
    else:
        with open(operationalC_file,'r') as f:
            operationalC_values = json.load(f)
        lifetime = operationalC_values['lifetime']


    with open(packageC_file,'r') as f:
        packageC_values = json.load(f)
    interposer_node = packageC_values['interposer_node']
    rdl_layer = packageC_values['rdl_layers']
    emib_layers = packageC_values['emib_layers']
    emib_pitch = packageC_values['emib_pitch']
    tsv_pitch = packageC_values['tsv_pitch']
    ubump_pitch = packageC_values['ubump_pitch']
    hyb_pitch = packageC_values['hyb_pitch']
    tsv_size = packageC_values['tsv_size']
    ubump_size = packageC_values['ubump_size']
    hyb_size = packageC_values['hyb_size']
    numBEOL = packageC_values['num_beol']

    
    result = calculate_CO2(design,scaling_factors, 'Tiger Lake',
                           num_iter,package_type=package_type ,Ns=num_prt_mfg,lifetime=lifetime,
                           carbon_per_kWh=carbon_per_kWh,transistors_per_gate=transistors_per_gate,
                           power_per_core=power_per_core,interposer_node = interposer_node, rdl_layer=rdl_layer, emib_layers=emib_layers,
                           emib_pitch=emib_pitch, tsv_pitch=tsv_pitch, ubump_pitch=ubump_pitch, hyb_pitch=hyb_pitch, tsv_size=tsv_size, ubump_size=ubump_size, hyb_size=hyb_size, 
                           num_beol=numBEOL)
    if print_info:
        print("'"+design_dir+"' Example testcase")
        print(" ---------------------------------------------------------")
        print("Manufacture Carbon in Kgs ")
        print(result[0]/1000) #Converting to Kgs
        print(" ---------------------------------------------------------")
        print("Design Carbon in Kgs ")
        print(result[1]/1000) #Converting to Kgs
        print(" ---------------------------------------------------------")
        print("Operational Carbon in Kgs ")
        print(result[3]/1000) #Converting to Kgs
        print(" ---------------------------------------------------------")
        print("Total Carbon in Kgs ")
        print(result[2]/1000) #Converting to Kgs
        print(" ---------------------------------------------------------")

    total_carbon = result[2].to_numpy().sum()
    total_mfgC = result[0].to_numpy().sum()
    total_desC = result[1].to_numpy().sum()
    total_opeC = result[3].to_numpy().sum()
    total_embC = total_mfgC + total_desC
    
    total_embC = total_embC * num_prt_mfg
    total_opeC = total_opeC * num_prt_mfg 
    total_carbon = total_embC + total_opeC
   
    if print_info:
        print(f"################################")
        print("Summary for the Chiplet : ")
        print(f"Total Emb C : {total_embC/1000} Kgs")
        print(f"Total Ope C : {total_opeC/1000} Kgs")
        print(f"Total Carbon : {total_carbon/1000} Kgs")
    ###############################
    return total_embC/1000, total_opeC/1000, total_carbon/1000 #in Kgs
