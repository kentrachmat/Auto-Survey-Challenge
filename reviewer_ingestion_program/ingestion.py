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
    vprint(VERBOSE,  "\n========== Ingestion program version " + str(VERSION) + " ==========\n") 
    vprint(VERBOSE,  "************************************************")
    vprint(VERBOSE,  "******** Processing dataset " + data_name.capitalize() + " ********")
    vprint(VERBOSE,  "************************************************")

    # Reading and converting data
    vprint(VERBOSE,  "========= Reading and converting data ==========")
    data, meta_data = read_data(input_dir)

    return data, meta_data


def process_data(data):
    reviewer_X = data["reviewer"]["papers"]
    
    return reviewer_X


def execute_model(reviewer_X, data):
    # Creating a model
    M = model()

    # Review papers
    vprint( VERBOSE,  "======== Reviewing papers ==========")
    reviewer_Y_hat = M.review_papers(reviewer_X, instruction=data["reviewer"]["instructions"])
    vprint( VERBOSE,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - OVERALL_START))

    return M, reviewer_Y_hat


def save_results(M, data_name, reviewer_Y_hat, output_dir):
    # Saving results
    filename_generator = data_name + f'_generator.predict'
    filename_reviewer = data_name + f'_reviewer.predict'
    vprint( VERBOSE, "======== Saving results to: " + output_dir)
    data_io.write(os.path.join(output_dir,filename_reviewer), reviewer_Y_hat)

    vprint( VERBOSE,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - OVERALL_START))
    time_spent = time.time() - OVERALL_START 
    time_left_over = MAX_TIME - time_spent
    vprint( VERBOSE,  "[+] End cycle, time left %5.2f sec" % time_left_over)

    del M, reviewer_Y_hat
    gc.collect()


def main(input_dir, output_dir, program_dir, submission_dir, data_name):
    data, meta_data = initialize(input_dir, output_dir, program_dir, submission_dir, data_name)
    reviewer_X = process_data(data)
    M, reviewer_Y_hat = execute_model(reviewer_X, data)
    save_results(M, data_name, reviewer_Y_hat, output_dir)

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
    
    if VERBOSE: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        print("Data name: "+ data_name)

    # Add path
    path.append(program_dir)
    path.append(submission_dir)

    # Local modules
    import data_io                      
    from data_io import vprint          
    from data_io import read_data
    from model import model
    
    # Move old results and create a new output directory
    if SAVE_PREV_RESULTS:
        data_io.mvdir(output_dir, output_dir+'_'+THE_DATE) 
    data_io.mkdir(output_dir) 
    
    # Show directory structure
    if DEBUG_MODE >= 4: 
        data_io.show_dir(".")
        
    main(input_dir, output_dir, program_dir, submission_dir, data_name)
