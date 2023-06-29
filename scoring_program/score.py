#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir scoring_output_dir

import os
import json
import time
from sys import argv

import libscores
from evaluator import Evaluator

# Set up default directories and file names:
ROOT_DIR = "../"
DEFAULT_SOLUTION_DIR = os.path.join(ROOT_DIR, "input_data")
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, "sample_result_submission")
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, "scoring_output")
DEFAULT_DATA_NAME = "ai_paper_challenge"

# Set debug and missing score defaults
DEBUG_MODE = 0
MISSING_SCORE = -0.999999

# Define scoring version
SCORING_VERSION = 1.0

def json_2_html(paper, html_file):
    for index, data in enumerate(paper):
        if index == 0:
            html_file.write(f"<h3>{data['heading']}: {data['text']}</h3><br><br>")
        else:
            html_file.write(f"<div>{data['heading']}<br>{data['text']}</div><br><br>")

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
    evaluator.read_solutions(solution_dir)
    generator_predict_file = os.path.join(prediction_dir, f'{data_name}_generator.predict')
    reviewer_predict_file = os.path.join(prediction_dir, f'{data_name}_reviewer.predict')
    evaluator.read_predictions(generator_predict_file, reviewer_predict_file)
    return evaluator.get_reviewer_scores(), evaluator.get_generator_scores()
    # except Exception as e:
    #     print(f"Error in scoring program: {e}")
    #     return MISSING_SCORE, MISSING_SCORE

def print_scores_each_paper(data, evaluator, html_file, predicted_data, prompt_name, html_comments):
    """ Print scores for each paper """
    html_file.write("<h3>Generator Score:</h3>")
    evaluator.write_json_score_to_html(data, html_file, html_comments, "each_paper")
    html_file.write("<b>================================</b><br>")
    html_file.write(f"<h3>Prompt : {prompt_name}</h3>")
    html_file.write("<b>================================</b><br>")
    json_2_html(json.loads(predicted_data), html_file)
    

def write_to_output_files(score_dir, generator_overall_score, generator_score, reviewer_score, evaluator, html_comments, duration):
    """ Write output results to JSON and HTML files """
    overall_generator_score, overall_reviewer_score = evaluator.get_overall_scores()

    for index, data in enumerate(generator_score):
        with open(os.path.join(score_dir, f'paper_{index}.html'), 'w') as html_file:
            print_scores_each_paper(data, evaluator, html_file, evaluator.generator_predictions[index], evaluator.generator_prompts[index], html_comments[index])

    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file, \
         open(os.path.join(score_dir, 'scores.html'), 'w') as html_file:
        
        html_file.write(f"<h2>======= AI Generator and Reviewer Results =======</h2>") 
        
        print_scores_generator("Generator Score", generator_overall_score, evaluator, overall_generator_score, html_file, index)
        print_scores_reviewer("Reviewer Score", reviewer_score, evaluator, html_file)
        evaluator.plot_bbox_differences_to_html(html_file)

        score_json = {
            'generator': overall_generator_score,
            'reviewer': overall_reviewer_score,
            'average': 0.5 * overall_generator_score + 0.5 * overall_reviewer_score,
            'duration': duration
        }
        score_file.write(json.dumps(score_json))

def print_scores_generator(score_title, score, evaluator, overall_generator_score, html_file, index):
    """ Print and write scores to HTML files """
    if score == MISSING_SCORE:
        print(f"======= {score_title}: ERROR =======")
        html_file.write(f"======= {score_title}: ERROR =======\n")
    else:
        print(f"======= {score_title}: =======\n{evaluator.convert_json_score_to_text(score)}================================\n")

        html_file.write(f"<strong>======= {score_title}: =======</strong><br>")

        html_file.write(f"<small>NOTE: if you have <span style='color:red;'>red</span> colored score, meaning that your corresponding score is below 0.5.</small>")
        html_file.write(f"<p>Link to each generated papers:</p>")

        html_file.write("<ol>")
        for i in range(index + 1):
            html_file.write(f"<li><a href='paper_{i}.html'>Detailed review for prompt {i}</a></li>")
        html_file.write("</ol>")

        html_file.write(f"<p>Average score:</p>")
        evaluator.write_json_score_to_html(score, html_file)
        html_file.write(f"<p>Overall score : {overall_generator_score:.2f}</p><br>")
        html_file.write("<strong>================================</strong><br>")
        html_file.write("<br>")


def print_scores_reviewer(score_title, score, evaluator, html_file):
    """ Print and write scores to HTML files """
    if score == MISSING_SCORE:
        print(f"======= {score_title}: ERROR =======")
        html_file.write(f"======= {score_title}: ERROR =======\n")
    else:
        print(f"======= {score_title}: =======\n{evaluator.convert_json_score_to_text(score)}================================\n")
        html_file.write(f"<strong>======= {score_title}: =======</strong><br>\n")
        evaluator.write_json_score_to_html(score, html_file)
        html_file.write("<strong>================================</strong><br>\n")
        html_file.write("<br>\n")


def main():
    """ Main function to coordinate scoring """
    start = time.time()
    solution_dir, prediction_dir, score_dir, data_name = process_arguments(argv)
    create_score_directory(score_dir)
    evaluator = Evaluator()

    reviewer_score, (generator_overall_score, generator_score, html_comments) = compute_scores(evaluator, solution_dir, prediction_dir, data_name)
    write_to_output_files(score_dir, generator_overall_score, generator_score, reviewer_score, evaluator, html_comments, duration=time.time() - start)

    if DEBUG_MODE > 1:
        libscores.show_platform()
        libscores.show_io(prediction_dir, score_dir)
        libscores.show_version(SCORING_VERSION)

    print(f'Scoring completed in {time.time() - start} seconds.')


if __name__ == "__main__":
    main()
