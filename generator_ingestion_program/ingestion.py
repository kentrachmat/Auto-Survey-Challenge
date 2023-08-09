#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# Dependencies
import gc
import glob
import numpy as np
import time
import os
from sys import argv, path
import datetime


# Configurations
VERBOSE = True 
DEBUG_MODE = 0
MAX_TIME = 500
MAX_CYCLE = 1
MAX_ESTIMATORS = float('Inf')
MAX_SAMPLES = 50000
SAVE_MODEL = True
SAVE_PREV_RESULTS = False
ROOT_DIR = "../"
DEFAULT_INPUT_DIR = ROOT_DIR + "input_data"
DEFAULT_OUTPUT_DIR = ROOT_DIR + "sample_result_submission"
DEFAULT_PROGRAM_DIR = ROOT_DIR + "ingestion_program"
DEFAULT_SUBMISSION_DIR = ROOT_DIR + "sample_code_submission"
DEFAULT_DATA_NAME = "ai_paper_challenge"
VERSION = 6 

# For tracking the time
OVERALL_START = time.time()         
THE_DATE = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

def initialize(input_dir, output_dir, program_dir, submission_dir, data_name):
    vprint(VERBOSE,  "************************************************")
    vprint(VERBOSE,  "**************** AI-AUTHOR track ***************")
    vprint(VERBOSE,  "************************************************")
    vprint(VERBOSE,  "\n============== Ingestion program  ==============\n") 
    

    # Reading and converting data
    vprint(VERBOSE,  "========= Reading data ==========")
    data, meta_data = read_data(input_dir, MIN_NUM_PROMPTS=NUM_PAPERS_TO_GENERATE)

    return data, meta_data


def process_data(data):
    generator_X = data["generator"]["prompts"]
    
    return generator_X


def execute_model(generator_X, data):
    # Creating a model
    M = model()

    # Generate papers
    vprint( VERBOSE,  "======== Generating papers ==========")
    generator_Y_hat = M.generate_papers(generator_X, instruction=data["generator"]["instructions"])
    vprint( VERBOSE,  "[+] Generation success, time spent so far %5.2f sec" % (time.time() - OVERALL_START))

    return M, generator_Y_hat


def save_results(M, data_name, generator_Y_hat, output_dir, ids=None):
    # Saving results
    filename_generator = data_name + f'_generator.predict'
    vprint( VERBOSE, "======== Saving results to: " + output_dir)
    data_io.write(os.path.join(output_dir,filename_generator), generator_Y_hat, ids=ids)

    vprint( VERBOSE,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - OVERALL_START))
    time_spent = time.time() - OVERALL_START 
    time_left_over = MAX_TIME - time_spent
    vprint( VERBOSE,  "[+] End cycle, time left %5.2f sec" % time_left_over)

    del M, generator_Y_hat
    gc.collect()


def main(input_dir, output_dir, program_dir, submission_dir, data_name):
    data, meta_data = initialize(input_dir, output_dir, program_dir, submission_dir, data_name)
    generator_X = process_data(data)
    M, generator_Y_hat = execute_model(generator_X, data)
    save_results(M, data_name, generator_Y_hat, output_dir, ids=data["generator"]["ids"])

    overall_time_spent = time.time() - OVERALL_START

    vprint( VERBOSE,  "[+] Done")
    vprint( VERBOSE,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % MAX_TIME)

    print("Ingestion program successfully completed.")

if __name__=="__main__" and DEBUG_MODE<4:	
    # Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = DEFAULT_INPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR
        program_dir= DEFAULT_PROGRAM_DIR
        submission_dir= DEFAULT_SUBMISSION_DIR
        data_name = DEFAULT_DATA_NAME
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
        data_name = DEFAULT_DATA_NAME
    
    # if VERBOSE: 
    #     print("Using input_dir: " + input_dir)
    #     print("Using output_dir: " + output_dir)
    #     print("Using program_dir: " + program_dir)
    #     print("Using submission_dir: " + submission_dir)
    #     print("Data name: "+ data_name)

    # Add path
    path.append(program_dir)
    path.append(submission_dir)

    # Local modules
    import data_io                      
    from data_io import vprint          
    from data_io import read_data

    try:
        from model import model
        from cfg import NUM_PAPERS_TO_GENERATE
    except:
        print("WARNING: No model.py found. Looking for model.py in one directory lower.")
        try:
            available_dirs = os.listdir(submission_dir)
            # add all directories to path
            for d in available_dirs:
                path.append(os.path.join(submission_dir, d))
            from model import model
            from cfg import NUM_PAPERS_TO_GENERATE
        except:
            print("ERROR: Could not find model.py. Please add model.py to the submission directory.")
            exit(0)
    
    # Move old results and create a new output directory
    if SAVE_PREV_RESULTS:
        data_io.mvdir(output_dir, output_dir+'_'+THE_DATE) 
    data_io.mkdir(output_dir) 
    
    # Show directory structure
    if DEBUG_MODE >= 4: 
        data_io.show_dir(".")
        
    main(input_dir, output_dir, program_dir, submission_dir, data_name)
