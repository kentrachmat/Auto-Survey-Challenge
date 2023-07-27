#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir scoring_output_dir

import os
import json
import time
from sys import argv
import libscores
from evaluator import Evaluator

from utils import *
from config import SEPARATE_HTML_EACH_PAPER

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

from config import DEBUG, CONTESTANT_MODE


# Input Processing
def create_score_directory(score_dir):
    """ Create scoring directory if it doesn't exist """
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

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


# Computing the scores
def compute_scores(evaluator, solution_dir, prediction_dir, data_name):
    """ Compute reviewer and generator scores """
    # try:
    generator_predict_file = os.path.join(prediction_dir, f'{data_name}_generator.predict')
    evaluator.read_generator_solutions_and_predictions(solution_dir, generator_predict_file)
    return evaluator.get_generator_scores()

# HTML Comments
def write_to_output_files(score_dir, generator_overall_score, generator_score, evaluator, html_comments, duration):
    """ Write output results to JSON and HTML files """
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file, \
         open(os.path.join(score_dir, 'scores_ai_author.html'), 'w') as html_file:

        html_file.write("<style>body {font-size: 16pt;}</style>\n")
        
        html_file.write(f"<center><h1>AI Author track</h1></center>")

        html_file.write("<center>To obtain the scores for the AI Reviewer track, see <a href=\"https://sites.google.com/view/ai-agents/faq?authuser=0#h.ec74x5ukwugy\">FAQ</a>.</center><br>")

        html_file.write(f"Below are the scores of your submission, obtained after AI-referee-reviewer review <b>{len(generator_score)}</b> papers.<br>")

        final_generator_overall_score = evaluator.get_overall_generator_scores() 
        
        print_scores_generator("Review Score", generator_overall_score, evaluator, final_generator_overall_score, html_file, num_prompts = len(generator_score))

        if not SEPARATE_HTML_EACH_PAPER:
            for index, data in enumerate(generator_score):
                html_file.write(f"<h2 id=\"paper_{index+1}\">Our assessment for Prompt {index + 1}:</h2><br>")
                print_scores_each_paper(data, evaluator, html_file, evaluator.generator_predictions[index], evaluator.generator_prompts[index], html_comments[index])


        score_json = {
            'score': final_generator_overall_score,
            'duration': duration
        }
        
        score_file.write(json.dumps(score_json))


    if SEPARATE_HTML_EACH_PAPER:
        # Make directory for full papers
        if not os.path.exists(os.path.join(score_dir, 'ai_author_full_papers')):
            os.makedirs(os.path.join(score_dir, 'ai_author_full_papers'))

        for index, data in enumerate(generator_score):
            with open(os.path.join(score_dir, 'ai_author_full_papers', f'paper_{index+1}.html'), 'w') as html_file:
                print_scores_each_paper(data, evaluator, html_file, evaluator.generator_predictions[index], evaluator.generator_prompts[index], html_comments[index])

def write_json_paper_to_html_file(json_paper, html_file):
    for index, data in enumerate(json_paper):
        if index == 0:
            html_file.write(f"<h3>{data['heading']}: {data['text']}</h3><br><br>")
        else:
            html_file.write(f"<div>{data['heading']}<br>{data['text']}</div><br><br>")

def print_scores_each_paper(data, evaluator, html_file, generated_paper, prompt_name, html_comments):
    """ Print scores for each generated paper """
    html_file.write("<h3>Review Score:</h3>")
    evaluator.write_json_score_to_html(data, html_file, html_comments, "each_paper")
    html_file.write("<b>================================</b><br>")
    html_file.write(f"<h3>Prompt : {prompt_name}</h3>")
    html_file.write("<b>================================</b><br>")

    """ Print the generated paper """
    print(f"======= Prompt : {prompt_name} =======")
    print(f"======= Generated Paper =======")
    print(generated_paper)
    json_paper = custom_json_loads(generated_paper)
    write_json_paper_to_html_file(json_paper, html_file)

def print_scores_generator(score_title, score, evaluator, overall_generator_score, html_file, num_prompts):
    """ Print and write scores to HTML files """
    if score == MISSING_SCORE:
        print(f"======= {score_title}: ERROR =======")
        html_file.write(f"======= {score_title}: ERROR =======\n")
    else:
        print(f"======= {score_title}: =======\n{evaluator.convert_json_score_to_text(score)}================================\n")
        # html_file.write(f"<br><small>Score based on rating good papers better than bad papers.</small><br>")
        html_file.write(f"<p>Detailed reviews:</p>")

        html_file.write("<ol>")
        for i in range(num_prompts):
            if SEPARATE_HTML_EACH_PAPER:
                html_file.write(f"<li><a href='ai_author_full_papers/paper_{i+1}.html'>Prompt {i+1}</a></li>")
            else:
                # Use anchor tag to link to the prompt
                html_file.write(f"<li><a href=\"#paper_{i+1}\">Prompt {i+1}</a></li>")
        html_file.write("</ol>")

        html_file.write(f"<p>Average score:</p>")
        html_file.write(f"<small><span style='color:red;'>Red</span> means score below 0.5.</small><br>\n")
        evaluator.write_json_score_to_html(score, html_file)
        html_file.write(f"<p>Overall score : {overall_generator_score:.2f}</p>")
        html_file.write(f"<small><small>Overall = relevance * (contribution + responsibility + clarity + soundness)/4</small></small><br>")
        html_file.write("<br>")

        if CONTESTANT_MODE:
            description = """
        <h1>Paper Evaluation Description</h1>

        <h2>Criterion: Relevance</h2>
        <p>Description:</p>
        <p>Does the answer address the question asked in the prompt?</p>

        <h2>Criterion: Contribution</h2>
        <p>Description:</p>
        <p>Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints?</p>

        <h2>Criterion: Responsibility</h2>
        <p>Description:</p>
        <p>Does the paper address potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy?</p>

        <h2>Criterion: Clarity</h2>
        <p>Description:</p>
        <p>Is the paper written in good English, with correct grammar, and precise vocabulary? Is the paper well organized in meaningful sections and subsections? Are the concepts clearly explained, with short sentences?</p>

        <h2>Criterion: Soundness</h2>
        <p>Description:</p>
        <p>Does the answer present accurate facts, supported by citations of authoritative references?</p>
        """
        else:
            description = """
        <h1>Paper Evaluation Description</h1>
        <h2>Criterion: Relevance</h2>
        <p>Description:</p>
        <p>Title: Does the title address the question asked in the prompt?</p>
        <p>Abstract: Does the abstract address the question asked in the prompt?</p>
        <p>Citations: Does the paper cite relevant papers that address the question asked in the prompt?</p>

        <h2>Criterion: Contribution</h2>
        <p>Description:</p>
        <p>Coverage: Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints?</p>
        <p>Abstract: How well does the abstract summarize the paper?</p>
        <p>Conclusion: How well does the conclusion highlight the main findings of the paper?</p>

        <h2>Criterion: Responsibility</h2>
        <p>Description:</p>
        <p>Does the paper address potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy, and is free of libelous or unlawful statements, does not infringe upon the rights of others, or contain material or instructions that might cause harm or injury?</p>

        <h2>Criterion: Clarity</h2>
        <p>Description:</p>
        <p>Correct Language: Is the paper written in good English, with correct grammar, and precise vocabulary?</p>
        <p>Organization: Is the paper well organized in meaningful sections and subsections?</p>
        <p>Explanations: Are the concepts clearly explained, with short sentences?</p>

        <h2>Criterion: Soundness</h2>
        <p>Description:</p>
        <p>C1 (references that exists) : the correctness of the references cited by the AI</p>
        <p>C2 (relevant references) : the relevance of the cited papers</p>
        <p>C3 (diverse references) : the interrelatedness of the cited papers</p>
        """
        html_file.write(description)


def main():
    """ Main function to coordinate scoring """
    start = time.time()
    print("\n============== Scoring program  ==============\n")
    solution_dir, prediction_dir, score_dir, data_name = process_arguments(argv)
    create_score_directory(score_dir)
    evaluator = Evaluator()

    generator_overall_score, generator_score, html_comments = compute_scores(evaluator, solution_dir, prediction_dir, data_name)
    write_to_output_files(score_dir, generator_overall_score, generator_score, evaluator, html_comments, duration=time.time() - start)

    if DEBUG_MODE > 1:
        libscores.show_platform()
        libscores.show_io(prediction_dir, score_dir)
        libscores.show_version(SCORING_VERSION)

    print(f'Scoring completed in {time.time() - start} seconds.')


if __name__ == "__main__":
    main()