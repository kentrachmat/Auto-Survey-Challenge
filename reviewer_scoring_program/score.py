#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir scoring_output_dir

import os
import glob
import json
import time
from sys import argv
from sklearn.metrics import accuracy_score

import libscores
import yaml
from evaluator import Evaluator

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


def compute_scores(evaluator, solution_dir, prediction_dir, data_name):
    """ Compute reviewer and generator scores """
    # try:
    evaluator.read_reviewer_solutions(solution_dir)
    reviewer_predict_file = os.path.join(prediction_dir, f'{data_name}_reviewer.predict')
    evaluator.read_reviewer_predictions(reviewer_predict_file)
    numeric_reviewer_scores = evaluator.get_numeric_reviewer_scores()
    good_paper_text_reviewer_scores = evaluator.get_good_paper_text_reviewer_scores()
    bad_paper_text_reviewer_scores = evaluator.get_bad_paper_text_reviewer_scores()
    three_highest_score_paper_details = evaluator.get_three_highest_score_paper_details()
    three_lowest_score_paper_details = evaluator.get_three_lowest_score_paper_details()
    return numeric_reviewer_scores, good_paper_text_reviewer_scores, bad_paper_text_reviewer_scores, three_highest_score_paper_details, three_lowest_score_paper_details


def write_to_output_files(score_dir, numeric_reviewer_scores, good_paper_text_reviewer_scores, bad_paper_text_reviewer_scores, three_highest_score_paper_details, three_lowest_score_paper_details, evaluator, duration):
    """ Write output results to JSON and HTML files """
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file, \
         open(os.path.join(score_dir, 'scores_ai_reviewer.html'), 'w') as html_file:

        # Set up default font size for the whole HTML file
        html_file.write("<style>body {font-size: 16pt;}</style>\n")

        html_file.write(f"<center><h1><b>AI-Reviewer track</b></h1></center><br>")
        html_file.write(f"<font size=\"+3\">Below are the scores of your submission, obtained by our meta-reviewer after reviewing <b>{evaluator.get_number_of_papers()}</b> papers</font><br>\n")


        html_file.write("<p>")
        print_numeric_scores("Average meta-reviewer evaluation of your reviewer's NUMERICAL scores", numeric_reviewer_scores, evaluator, html_file)
        
        html_file.write(f"<h2>======= Average meta-reviewer evaluation of your reviewer's TEXT comments =======</h2>")
        
        html_file.write("Three best meta-reviews:<br>\n")
        html_file.write("<ol>")
        for i in range(3):
            html_file.write(f"<li><a href='ai_reviewer_full_papers/best_reviews_paper_{i+1}.html'>Paper {i+1}</a></li>")
        html_file.write("</ol>")

        html_file.write("Three worst meta-reviews:<br>\n")
        html_file.write("<ol>")
        for i in range(3):
            html_file.write(f"<li><a href='ai_reviewer_full_papers/worst_reviews_paper_{i+1}.html'>Paper {i+1}</a></li>")
        html_file.write("</ol>")
        html_file.write("<br>")
        
        html_file.write("Average evaluation of GOOD papers:<br><br>")
        evaluator.plot_reviewer_scores_to_html(good_paper_text_reviewer_scores, html_file)
        html_file.write("<br>\n")
        html_file.write("Average evaluation of BAD papers:<br><br>")
        evaluator.plot_reviewer_scores_to_html(bad_paper_text_reviewer_scores, html_file)


        html_file.write("<strong>================================</strong><br>")

        overall_reviewer_score = evaluator.get_overall_reviewer_scores()
        score_json = {
            'score': overall_reviewer_score,
            'duration': duration
        }
        score_file.write(json.dumps(score_json))
        
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
        html_file.write(f"<h2>======= {score_title} =======</h2>")

        evaluator.write_numeric_json_score_to_html(score, html_file)

        html_file.write(f"<br><small>Score based on rating good papers better than bad papers.</small><br>")
        html_file.write(f"<small><span style='color:red;'>Red</span> means score below 0.5.</small><br>\n")
        html_file.write("<br>\n")

def main():
    """ Main function to coordinate scoring """
    start = time.time()
    solution_dir, prediction_dir, score_dir, data_name = process_arguments(argv)
    create_score_directory(score_dir)
    evaluator = Evaluator()

    numeric_reviewer_scores, good_paper_text_reviewer_scores, bad_paper_text_reviewer_scores, three_highest_score_paper_details, three_lowest_score_paper_details = compute_scores(evaluator, solution_dir, prediction_dir, data_name)
    write_to_output_files(score_dir, numeric_reviewer_scores, good_paper_text_reviewer_scores, bad_paper_text_reviewer_scores, three_highest_score_paper_details, three_lowest_score_paper_details, evaluator, duration=time.time() - start)

    if DEBUG_MODE > 1:
        libscores.show_platform()
        libscores.show_io(prediction_dir, score_dir)
        libscores.show_version(SCORING_VERSION)

    print(f'Scoring completed in {time.time() - start} seconds.')


if __name__ == "__main__":
    main()
