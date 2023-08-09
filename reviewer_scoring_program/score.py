#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir scoring_output_dir

import os
import json
import time
from sys import argv
from metacriteria.utils import custom_json_loads
import libscores
from evaluator import Evaluator
from config import SEPARATE_HTML_EACH_PAPER
import threading

numeric_reviewer_scores = None
good_paper_text_reviewer_scores = None
bad_paper_text_reviewer_scores = None

# Set up default directories and file names:
ROOT_DIR = "../"
DEFAULT_SOLUTION_DIR = os.path.join(ROOT_DIR, "sample_data")
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, "sample_result_submission")
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, "scoring_output")
DEFAULT_DATA_NAME = "ai_paper_challenge"

# Set debug and missing score defaults
DEBUG_MODE = 0
MISSING_SCORE = -0.999999

# Define scoring version
SCORING_VERSION = 1.0

def process_arguments(arguments):
    """ Process command line arguments """
    if len(arguments) == 1:  # Default directories
        return DEFAULT_SOLUTION_DIR, DEFAULT_PREDICTION_DIR, DEFAULT_SCORE_DIR, DEFAULT_DATA_NAME
    elif len(arguments) == 3: # Codalab's default configuration
        solution_dir = os.path.join(arguments[1], 'ref')
        prediction_dir = os.path.join(arguments[1], 'res')
        return solution_dir, prediction_dir, arguments[2], DEFAULT_DATA_NAME
    elif len(arguments) == 4: # User-specified directories
        return arguments[1], arguments[2], arguments[3], DEFAULT_DATA_NAME
    else:
        raise ValueError('Wrong number of arguments passed.')

def create_score_directory(score_dir):
    """ Create scoring directory if it doesn't exist """
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)


def get_numeric_reviewer_scores(evaluator):
    global numeric_reviewer_scores
    numeric_reviewer_scores = evaluator.get_generic_reviewer_scores("key1")

def get_good_paper_text_reviewer_scores(evaluator):
    global good_paper_text_reviewer_scores
    good_paper_text_reviewer_scores = evaluator.get_generic_reviewer_scores("key2")

def get_bad_paper_text_reviewer_scores(evaluator):
    global bad_paper_text_reviewer_scores
    bad_paper_text_reviewer_scores = evaluator.get_generic_reviewer_scores("key3")


def compute_scores(evaluator, solution_dir, prediction_dir, data_name):
    """ Compute reviewer and generator scores """
    # try:
    reviewer_predict_file = os.path.join(prediction_dir, f'{data_name}_reviewer.predict')
    evaluator.read_reviewer_solutions_and_predictions(solution_dir, reviewer_predict_file)

    thread_numeric_scores = threading.Thread(target=get_numeric_reviewer_scores, args=(evaluator,))
    thread_good_scores = threading.Thread(target=get_good_paper_text_reviewer_scores, args=(evaluator,))
    thread_bad_scores = threading.Thread(target=get_bad_paper_text_reviewer_scores, args=(evaluator,))

    thread_numeric_scores.start()
    thread_good_scores.start()
    thread_bad_scores.start()

    thread_numeric_scores.join()
    thread_good_scores.join()
    thread_bad_scores.join()

    # numeric_reviewer_scores = evaluator.get_numeric_reviewer_scores()
    # good_paper_text_reviewer_scores = evaluator.get_good_paper_text_reviewer_scores()
    # bad_paper_text_reviewer_scores = evaluator.get_bad_paper_text_reviewer_scores()

    three_highest_score_paper_details = evaluator.get_three_highest_score_paper_details()
    three_lowest_score_paper_details = evaluator.get_three_lowest_score_paper_details()
    return numeric_reviewer_scores, good_paper_text_reviewer_scores, bad_paper_text_reviewer_scores, three_highest_score_paper_details, three_lowest_score_paper_details


def write_to_output_files(score_dir, numeric_reviewer_scores_output, good_paper_text_reviewer_scores_output, bad_paper_text_reviewer_scores_output, three_highest_score_paper_details, three_lowest_score_paper_details, evaluator, duration):
    """ Write output results to JSON and HTML files """
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file, \
         open(os.path.join(score_dir, 'scores_ai_reviewer.html'), 'w') as html_file:

        # Set up default font size for the whole HTML file
        html_file.write("<style>body {font-size: 16pt;}</style>\n")

        html_file.write(f"<center><h1><b>AI-Reviewer track</b></h1></center><br>")
        html_file.write(f"<font size=\"+3\">Below are the scores of your submission, obtained by our meta-reviewer after reviewing <b>{evaluator.get_number_of_papers()}</b> papers</font><br>\n")


        html_file.write("<p>")

        print_numeric_scores("Average meta-reviewer evaluation of your reviewer's NUMERICAL scores", numeric_reviewer_scores, evaluator, html_file)

        if evaluator.EVALUATION_MODE == 'full':
            
            
            html_file.write(f"<h2>Average meta-reviewer evaluation of your reviewer's TEXT comments:</h2><br>")
            
            # a small note The numbers below are the accuracy of rating good papers better than bad papers
            # html_file.write("<small>The numbers below are the accuracy of rating good papers better than bad papers</small><br>")

            html_file.write("Three best meta-reviews:<br>\n")
            html_file.write("<ol>")
            for i in range(3):
                if SEPARATE_HTML_EACH_PAPER:
                    html_file.write(f"<li><a href='ai_reviewer_full_papers/best_reviews_paper_{i+1}.html'>Paper {i+1}</a></li>")
                else:
                    html_file.write(f"<li><a href='#best_reviews_paper_{i+1}'>Paper {i+1}</a></li>")
            html_file.write("</ol>")

            html_file.write("Three worst meta-reviews:<br>\n")
            html_file.write("<ol>")
            for i in range(3):
                if SEPARATE_HTML_EACH_PAPER:
                    html_file.write(f"<li><a href='ai_reviewer_full_papers/worst_reviews_paper_{i+1}.html'>Paper {i+1}</a></li>")
                else:
                    html_file.write(f"<li><a href='#worst_reviews_paper_{i+1}'>Paper {i+1}</a></li>")
            html_file.write("</ol>")
            html_file.write("<br>")
            
            html_file.write("Average evaluation of GOOD papers:<br><br>")
            evaluator.plot_reviewer_scores_to_html(good_paper_text_reviewer_scores, html_file)
            html_file.write("<br>\n")
            html_file.write("Average evaluation of BAD papers:<br><br>")
            evaluator.plot_reviewer_scores_to_html(bad_paper_text_reviewer_scores, html_file)


            if not SEPARATE_HTML_EACH_PAPER:
                for i in range(3):
                    html_file.write(f"<h2 id='best_reviews_paper_{i+1}'>Best meta-reviews {i+1}</h2>")
                    evaluator.write_paper_details_to_html_file(three_highest_score_paper_details[i], html_file, anchor_name=f'best_reviews_paper_{i+1}')
                for i in range(3):
                    html_file.write(f"<h2 id='worst_reviews_paper_{i+1}'>Worst meta-reviews {i+1}</h2>")
                    evaluator.write_paper_details_to_html_file(three_lowest_score_paper_details[i], html_file, anchor_name=f'worst_reviews_paper_{i+1}')




        overall_reviewer_score = evaluator.get_overall_reviewer_scores()
        score_json = {
            'score': overall_reviewer_score,
            'duration': duration
        }
        score_file.write(json.dumps(score_json))
        
    if SEPARATE_HTML_EACH_PAPER:
        # Make directory for paper details
        paper_details_dir = os.path.join(score_dir, 'ai_reviewer_full_papers')
        if not os.path.exists(paper_details_dir):
            os.makedirs(paper_details_dir)

        for i in range(3):
            with open(os.path.join(score_dir, 'ai_reviewer_full_papers', f'best_reviews_paper_{i+1}.html'), 'w') as html_file:
                evaluator.write_paper_details_to_html_file(three_highest_score_paper_details[i], html_file)

        for i in range(3):
            with open(os.path.join(score_dir, 'ai_reviewer_full_papers', f'worst_reviews_paper_{i+1}.html'), 'w') as html_file:
                evaluator.write_paper_details_to_html_file(three_lowest_score_paper_details[i], html_file)


def print_numeric_scores(score_title, score, evaluator, html_file):
    """ Print and write scores to HTML files """
    if score == MISSING_SCORE:
        print(f"======= {score_title}: ERROR =======")
        html_file.write(f"======= {score_title}: ERROR =======\n")
    else:
        print(f"======= {score_title}: =======\n{evaluator.convert_json_score_to_text(score)}================================\n")
        html_file.write(f"<h2>{score_title}:</h2>")

        html_file.write(f"<small><small>The scores below represent for each criterion the accuracy of rating good papers better than bad papers</small></small><br>")

        evaluator.write_numeric_json_score_to_html(score, html_file)

        html_file.write(f"<small><span style='color:red;'>Red</span> means score below 0.5.</small><br>\n")
        html_file.write("<br>")
        # html_file.write("To visualize how well your reviewer differentiated between good and bad papers, we plot the diffence between every pair of good and bad papers below:<br>")
        # The sentence above is too long, write a better one
        html_file.write("Box plot of score diffence between good and bad papers:<br>")
        html_file.write("<small><small>Each dot represents one pair of good and bad papers. The horizontal axis has no information.</small></small><br>")
        evaluator.plot_difference_of_scores_to_html(html_file)
        # Write a note in smaller text
        # html_file.write("<br><small>Horizontal axis has no information</small><br>")

def main():
    """ Main function to coordinate scoring """
    start = time.time()
    print("\n============== Scoring program  ==============\n")
    solution_dir, prediction_dir, score_dir, data_name = process_arguments(argv)
    create_score_directory(score_dir)
    evaluator = Evaluator()

    numeric_reviewer_scores_output, good_paper_text_reviewer_scores_output, bad_paper_text_reviewer_scores_output, three_highest_score_paper_details, three_lowest_score_paper_details = compute_scores(evaluator, solution_dir, prediction_dir, data_name)
    write_to_output_files(score_dir, numeric_reviewer_scores_output, good_paper_text_reviewer_scores_output, bad_paper_text_reviewer_scores_output, three_highest_score_paper_details, three_lowest_score_paper_details, evaluator, duration=time.time() - start)

    if DEBUG_MODE > 1:
        libscores.show_platform()
        libscores.show_io(prediction_dir, score_dir)
        libscores.show_version(SCORING_VERSION)

    print(f'Scoring completed in {time.time() - start} seconds.')


if __name__ == "__main__":
    main()
