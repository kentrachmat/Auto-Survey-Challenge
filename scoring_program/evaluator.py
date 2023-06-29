import os
from os.path import isfile
import numpy as np
import pandas as pd
import json5 as json
import random
import matplotlib.pyplot as plt
import base64
from scipy import stats
import re

from collections import defaultdict
from libscores import kendall_tau, safe_kendalltau

from baseline_reviewer import BaselineReviewer

class Evaluator:
    def __init__(self):
        self.solution_dir = None

        #Generator
        self.generator_prompts = None
        self.generator_predictions = None
        self.generator_solutions = None
        self.super_categories = ['clarity', 'contribution', 'soundness', 'responsibility', 'overall', 'confidence']
        self.generator_scores = None
        self.generator_score_html = []
        self.overall_generator_score = None

        # Reviewer
        self.reviewer_predictions = None
        self.reviewer_solutions = None
        self.reviewer_score_pairs = None
        self.overall_reviewer_scores = None

        # Baseline Reviewer
        self.baseline_reviewer = BaselineReviewer()


    def read_solutions(self, solution_dir):
        ''' Function to read the Labels from CSV files'''

        self.solution_dir = solution_dir

        print("###-------------------------------------###")
        print("### Checking Data")
        print("###-------------------------------------###\n\n")

        #----------------------------------------------------------------
        # Settings
        #----------------------------------------------------------------
        GENERATOR_PATH = os.path.join(solution_dir, "generator")
        REVIEWER_PATH = os.path.join(solution_dir, "reviewer")


        #----------------------------------------------------------------
        # Errors
        #----------------------------------------------------------------

        # Check Generator Directory
        if not os.path.exists(GENERATOR_PATH):
            print('[-] Generation prompts directory Not Found')
            return

        #Check Reviewer Directory
        if not os.path.exists(REVIEWER_PATH):
            print('[-] Essays for Reviewer Not Found')
            return


        #----------------------------------------------------------------
        # Load CSV
        #----------------------------------------------------------------


        print("###-------------------------------------###")
        print("### Loading Data")
        print("###-------------------------------------###\n\n")
        generator_df = pd.read_csv(os.path.join(GENERATOR_PATH, "prompts.csv"))
        reviewer_df = pd.read_csv(os.path.join(REVIEWER_PATH, "metadata.csv"))


        # Generator
        self.generator_prompts = generator_df['prompt'].values

        self.generator_solutions = []
        for paper_id in generator_df['id'].values:
            paraphrased_papers = {'good': [], 'bad': {}}
            for paper_filename in [x for x in os.listdir(os.path.join(GENERATOR_PATH, "papers", str(paper_id))) if x!= ".DS_Store"]:
                paper_text = open(os.path.join(GENERATOR_PATH, "papers", str(paper_id), paper_filename), 'r').read()
                if paper_filename.startswith("good"):
                    paraphrased_papers['good'].append(paper_text)
                elif paper_filename.startswith("bad"):
                    for super_category in self.super_categories:
                        if paper_filename.startswith("bad_" + super_category):
                            if paper_filename.startswith("bad_" + super_category + "_"):
                                if super_category not in paraphrased_papers['bad']:
                                    paraphrased_papers['bad'][super_category] = []
                                paraphrased_papers['bad'][super_category].append(paper_text)
                                break
                            else:
                                if super_category not in paraphrased_papers['bad']:
                                    paraphrased_papers['bad'][super_category] = defaultdict(list)
                                sub_category = paper_filename.split("bad_" + super_category, 1)[1].split("_")[0]
                                paraphrased_papers['bad'][super_category][sub_category].append(paper_text)
                                break
            self.generator_solutions.append(paraphrased_papers)
        
        # Reviewer
        self.reviewer_solutions = reviewer_df[['id', 'pdf_name', 'good_or_bad']]
        # for i in range(len(reviewer_df)):
        #     reviewer_solution = {}
        #     for criterion in self.reviewer_criteria:
        #         reviewer_solution[criterion] = reviewer_df[criterion][i]
        #     self.reviewer_solutions.append(reviewer_solution)
        
        print("###-------------------------------------###")
        print("### Solutions files are ready!")
        print("###-------------------------------------###\n\n")


    def read_predictions(self, generator_predict_file, reviewer_predict_file):
        if not os.path.isfile(generator_predict_file):
            print("#--ERROR--# Generator prediction file not found: " + generator_predict_file)
            raise ValueError("Generator prediction file not found: " + generator_predict_file)
        if not os.path.isfile(reviewer_predict_file):
            print("#--ERROR--# Reviewer prediction file not found: " + reviewer_predict_file)
            raise ValueError("Reviewer prediction file not found: " + reviewer_predict_file)

        # Read the solution and prediction values into list
        with open(generator_predict_file, 'r') as f:
            self.generator_predictions = f.read().split('\n\n\n\n')[:-1]
        with open(reviewer_predict_file, 'r') as f:
            self.reviewer_predictions = f.read().split('\n\n\n\n')[:-1]

        # use double quotes instead of single quotes
        loaded_reviewer_predictions = []
        for x in self.reviewer_predictions:
            try:
                loaded_reviewer_predictions.append(json.loads(x.replace("'", '"')))
            except:
                # Add a comma after every "}" if THE COMMA IS NOT THERE
                x = re.sub(r'}(?!\s*,)', r'},', x)
                while not x.endswith("}"):
                    x = x[:-1]
                loaded_reviewer_predictions.append(json.loads(x.replace("'", '"')))

        self.reviewer_predictions = loaded_reviewer_predictions
        # remove all spaces from the keys and keys of values of each of reviewer prediction
        for i in range(len(self.reviewer_predictions)):
            new_prediction = {}
            for key, value in self.reviewer_predictions[i].items():
                new_key = key.lower().replace(" ", "")
                if isinstance(value, dict):
                    new_value = {}
                    for sub_key, sub_value in value.items():
                        new_sub_key = sub_key.lower().replace(" ", "")
                        new_value[new_sub_key] = sub_value
                else:
                    new_value = value
                new_prediction[new_key] = new_value
            self.reviewer_predictions[i] = new_prediction


        if (len(self.generator_solutions) != len(self.generator_predictions)): 
            print("#--ERROR--# Number of lines in solution file (" + str(len(self.generator_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.generator_predictions)) + ")")
            raise ValueError("Number of lines in solution file (" + str(len(self.generator_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.generator_predictions)) + ")")
        if (len(self.reviewer_solutions) != len(self.reviewer_predictions)):
            print("#--ERROR--# Number of lines in solution file (" + str(len(self.reviewer_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.reviewer_predictions)) + ")")
            raise ValueError("Number of lines in solution file (" + str(len(self.reviewer_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.reviewer_predictions)) + ")")
        
    def get_generator_scores(self):
        ''' Function to get the overall score for generator'''
        if self.overall_generator_score is None:
            self.compute_generator_scores()
        return self.overall_generator_score, self.generator_scores, self.generator_score_html

    def compute_generator_scores(self):
        ''' Function to compute the score for generator'''

        # Compute the score for each criterion
        self.generator_scores = []
        for i in range(len(self.generator_solutions)):
            generator_score = {}
            prediction_prompt = self.generator_prompts[i]
            prediction_paper = self.generator_predictions[i]
            solution_papers = self.generator_solutions[i]
            good_papers = solution_papers['good']
            bad_papers = solution_papers['bad'] 

            available_criteria = defaultdict(dict)
            for super_category, super_value in solution_papers['bad'].items():
                if isinstance(super_value, list):
                    available_criteria[super_category] = "[0 if paper 1 is better, 1 if paper 2 is better]"
                else:
                    for sub_category in super_value:
                        available_criteria[super_category][sub_category] = "[0 if paper 1 is better, 1 if paper 2 is better]"

            generator_score = self.baseline_reviewer.compare_papers(good_papers, bad_papers, prediction_paper, prediction_prompt, available_criteria)
            # generator_score = {'contribution': {'conclusion': 0.5833333333333333, 'abstract': 0.75, 'title': 0.75, 'coverage': 0.16666666666666666}, 'responsibility': 0.16666666666666666, 'clarity': {'explanations': 0.6666666666666666, 'correctlanguage': 0.16666666666666666, 'organization': 0.16666666666666666}, 'soundness': {'c1': 0.9, 'c2': 0.8, 'c3': 0.9}, 'confidence': 0.8, 'overall': 0.8}
            html_comment = self.baseline_reviewer.get_html_comments(generator_score, prediction_paper)

            self.generator_scores.append(generator_score)
            self.generator_score_html.append(html_comment)


        # Combine the scores accross all papers
        self.overall_generator_score = defaultdict(lambda: defaultdict(int))
        for generator_score in self.generator_scores:
            for super_category, super_value in generator_score.items():
                if isinstance(super_value, dict):
                    for sub_category, sub_value in super_value.items():
                        self.overall_generator_score[super_category][sub_category] += sub_value/len(self.generator_scores)
                else:
                    if super_category not in self.overall_generator_score:
                        self.overall_generator_score[super_category] = 0
                    self.overall_generator_score[super_category] += super_value/len(self.generator_scores)        

    def get_reviewer_scores(self):
        """Get the reviewer scores for each criterion"""
        if self.overall_reviewer_scores is None:
            self.compute_reviewer_scores()
        return self.overall_reviewer_scores 

    def compute_reviewer_scores(self):
        """Compute the reviewer scores."""
        
        self.reviewer_score_pairs = defaultdict(dict)
        for solution_type in self.reviewer_solutions['good_or_bad'].unique():
            if 'good' in solution_type or 'human' in solution_type:
                continue
            criterion = solution_type.split("_")[1]
            for super_category in self.super_categories:
                if criterion.startswith(super_category):
                    if criterion == super_category:
                        self.reviewer_score_pairs[criterion] = []
                    else:
                        sub_category = criterion.split(super_category)[1]
                        self.reviewer_score_pairs[super_category][sub_category] = []
                    break

        # loop over all the criteria to get the score pairs
        good_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: 'good' in x)]
        good_scores_ids_as_index = good_scores_df.groupby('pdf_name')['id'].apply(list)
        good_scores_ids = good_scores_ids_as_index.values

        for criterion in self.reviewer_score_pairs:
            if isinstance(self.reviewer_score_pairs[criterion], list):
                bad_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: criterion in x)]
                bad_scores_ids_as_index = bad_scores_df.groupby('pdf_name')['id'].apply(list)
                bad_scores_ids_as_index = bad_scores_ids_as_index[good_scores_ids_as_index.index]
                bad_scores_ids = bad_scores_ids_as_index.values

                average_good_predictions = []
                for i in range(len(good_scores_ids)):
                    average_good_predictions.append(np.mean(
                        [self.reviewer_predictions[j][criterion] for j in good_scores_ids[i]]
                    ))
                    
                average_bad_predictions = []
                for i in range(len(bad_scores_ids)):
                    average_bad_predictions.append(np.mean(
                        [self.reviewer_predictions[j][criterion] for j in bad_scores_ids[i]]
                        ))
                self.reviewer_score_pairs[criterion] = list(zip(average_good_predictions, average_bad_predictions))
            else:
                for sub_category in self.reviewer_score_pairs[criterion]:
                    bad_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: sub_category in x)]
                    bad_scores_ids_as_index = bad_scores_df.groupby('pdf_name')['id'].apply(list)
                    bad_scores_ids_as_index = bad_scores_ids_as_index[good_scores_ids_as_index.index]
                    bad_scores_ids = bad_scores_ids_as_index.values

                    average_good_predictions = []
                    for i in range(len(good_scores_ids)):
                        average_good_predictions.append(np.mean(
                            [self.reviewer_predictions[j][criterion][sub_category] for j in good_scores_ids[i]]
                            ))
                    average_bad_predictions = []
                    for i in range(len(bad_scores_ids)):
                        average_bad_predictions.append(np.mean(
                            [self.reviewer_predictions[j][criterion][sub_category] for j in bad_scores_ids[i]]
                            ))
                    self.reviewer_score_pairs[criterion][sub_category] = list(zip(average_good_predictions, average_bad_predictions))

        # print(self.reviewer_score_pairs)

        # Calculate the t-statistic and p-value for each criterion
        self.overall_reviewer_scores = {}
        for criterion, score_pairs in self.reviewer_score_pairs.items():
            if isinstance(score_pairs, list):
                self.overall_reviewer_scores[criterion] = stats.ttest_rel([x[0] for x in score_pairs], [x[1] for x in score_pairs])[0]
            else:
                self.overall_reviewer_scores[criterion] = {}
                for sub_category, sub_score_pairs in score_pairs.items():
                    self.overall_reviewer_scores[criterion][sub_category] = stats.ttest_rel([x[0] for x in sub_score_pairs], [x[1] for x in sub_score_pairs])[0]
    
    def get_overall_scores(self):
        """Get the overall scores."""
        if self.overall_generator_score is None:
            self.compute_generator_scores()
        if self.overall_reviewer_scores is None:
            self.compute_reviewer_scores()
        generator_score = 0
        for super_category, super_value in self.overall_generator_score.items():
            if isinstance(super_value, dict):
                for sub_category, sub_value in super_value.items():
                    generator_score += sub_value / ( len(super_value) * len(self.overall_generator_score) )
            else:
                generator_score += super_value / len(self.overall_generator_score)
        
        reviewer_score = 0
        for super_category, super_value in self.overall_reviewer_scores.items():
            if isinstance(super_value, dict):
                for sub_category, sub_value in super_value.items():
                    reviewer_score += sub_value / ( len(super_value) * len(self.overall_reviewer_scores) )
            else:
                reviewer_score += super_value / len(self.overall_reviewer_scores)
        
        return generator_score, reviewer_score
    
    def convert_json_score_to_text(self, json_score):
        """Convert the json score to text."""
        text_score = ""
        for super_category, super_value in json_score.items():
            if isinstance(super_value, dict):
                text_score += f"{super_category}:\n"
                for sub_category, sub_value in super_value.items():
                    text_score += f"\t{sub_category}: {sub_value:.2f}\n"
            else:
                text_score += f"{super_category}: {super_value:.2f}\n"
        return text_score

    def write_json_score_to_html(self, json_score, html_file, html_comments="", types=None):
        """Convert the json score to text."""
        for super_category, super_value in json_score.items():
            if isinstance(super_value, dict):
                html_file.write(f"{super_category}:<br>\n")
                for sub_category, sub_value in super_value.items():
                    color = 'red' if sub_value < 0.5 else 'black'
                    if types is None:
                        html_file.write(f"&emsp;{sub_category}: <span style='color:{color};'>{sub_value:.2f}</span><br>")
                    else:
                        reason = "" if html_comments == "" else html_comments[super_category][sub_category]
                        html_file.write(f"&emsp;{sub_category}: <span style='color:{color};'>{sub_value:.2f}</span><br>&emsp;&emsp;reason: {reason}<br><br>")
            else:
                color = 'red' if super_value < 0.5 else 'black'
                if types is None:
                    html_file.write(f"{super_category}: <span style='color:{color};'>{super_value:.2f}</span><br>")
                else:
                    reason = "" if html_comments == "" else html_comments[super_category]
                    html_file.write(f"{super_category}: <span style='color:{color};'>{super_value:.2f}</span><br>&emsp;reason: {reason}<br><br>")

    def plot_bbox_differences_to_html(self, html_file):
        """Plot the bbox plot of the differences between good and bad scores for reviewer to html file."""
        for criterion in self.reviewer_score_pairs:
            if isinstance(self.reviewer_score_pairs[criterion], list):
                filepath = os.path.join('reviewer_bbox_difference_' + criterion + '.png')
                good_predictions = [x[0] for x in self.reviewer_score_pairs[criterion]]
                bad_predictions = [x[1] for x in self.reviewer_score_pairs[criterion]]
                
                t_statistic, p_value = stats.ttest_rel(good_predictions, bad_predictions)
                # plot the bbox plot of the differences between good and bad scores
                differences = np.array(good_predictions) - np.array(bad_predictions)
                bp = plt.boxplot(differences, showfliers=False)
                # Add some random "jitter" to the x-axis
                x = np.random.normal(1, 0.04, size=len(differences))
                plt.plot(x, differences, 'r.', alpha=0.5)
                # remove x-axis ticks
                plt.xticks([])
                plt.ylabel('Differences')
                plt.ylim([-1, 1])
                text = f"t-statistic: {t_statistic:.2f}\np-value: {p_value:.2e}"
                plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
                plt.savefig(filepath)
                plt.close()

                # Save the plot in the html file
                binary_fc = open(filepath, 'rb').read()
                base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
                ext = filepath.split('.')[-1]
                dataurl = f'data:image/{ext};base64,{base64_utf8_str}'

                html_file.write(f"<strong>{criterion}:</strong><br>\n")
                html_file.write("<img src="+dataurl+" alt='Difference plot' width='250'/>\n")
                html_file.write("<br>\n")
                os.remove(filepath)
            else:
                for sub_category in self.reviewer_score_pairs[criterion]:
                    filepath = os.path.join('reviewer_bbox_difference_' + criterion + '_' + sub_category + '.png')
                    good_predictions = [x[0] for x in self.reviewer_score_pairs[criterion][sub_category]]
                    bad_predictions = [x[1] for x in self.reviewer_score_pairs[criterion][sub_category]]
                    
                    t_statistic, p_value = stats.ttest_rel(good_predictions, bad_predictions)
                    # plot the bbox plot of the differences between good and bad scores
                    differences = np.array(good_predictions) - np.array(bad_predictions)
                    bp = plt.boxplot(differences, showfliers=False)
                    # Add some random "jitter" to the x-axis
                    x = np.random.normal(1, 0.04, size=len(differences))
                    plt.plot(x, differences, 'r.', alpha=0.5)
                    # remove x-axis ticks
                    plt.xticks([])
                    plt.ylabel('Differences')
                    plt.ylim([-1, 1])
                    text = f"t-statistic: {t_statistic:.2f}\np-value: {p_value:.2e}"
                    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=14,
                            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
                    plt.savefig(filepath)
                    plt.close()

                    # Save the plot in the html file
                    binary_fc = open(filepath, 'rb').read()
                    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
                    ext = filepath.split('.')[-1]
                    dataurl = f'data:image/{ext};base64,{base64_utf8_str}'

                    html_file.write(f"<strong>{criterion} {sub_category}:</strong><br>\n")
                    html_file.write("<img src="+dataurl+" alt='Difference plot' width='250'/>\n")
                    html_file.write("<br>\n")
                    os.remove(filepath)
        
