import os
from enum import Enum
from collections import defaultdict
import pandas as pd
import json5 as json
from tqdm.auto import tqdm
from nltk import sent_tokenize, word_tokenize
from config import DEBUG, CONTESTANT_MODE, USE_OUR_BASELINE_REVIEWER 

if USE_OUR_BASELINE_REVIEWER:
    from baseline_reviewer_referee import BaselineReviewer
else:
    from baseline_reviewer_chatgpt import BaselineReviewer

import numpy as np
from utils import *

class SuperCategories(Enum):
    RELEVANCE = 'relevance'
    CLARITY = 'clarity'
    CONTRIBUTION = 'contribution'
    RESPONSIBILITY = 'responsibility'
    # CONFIDENCE = 'confidence',
    SOUNDNESS = 'soundness'

class Evaluator:
    def __init__(self):
        self.solution_dir = None

        #Generator
        self.selected_IDs = None
        self.generator_prompts = None
        self.generator_predictions = None
        self.generator_solutions = None
        self.generator_html_comments = []

        # don't add bad Soundness paper for now
        self.super_categories = [category.value for category in SuperCategories]
        self.generator_scores = None
        self.overall_generator_score = None

        # Baseline Reviewer
        self.baseline_reviewer = BaselineReviewer()

    def read_generator_solutions_and_predictions(self, solution_dir, generator_predict_file):
        ''' Function to read the Labels from CSV files'''

        self.solution_dir = solution_dir

        print("###-------------------------------------###")
        print("### Checking Data")
        print("###-------------------------------------###\n\n")

        #----------------------------------------------------------------
        # Settings
        #----------------------------------------------------------------
        GENERATOR_PATH = os.path.join(solution_dir, "generator")

        #----------------------------------------------------------------
        # Errors
        #----------------------------------------------------------------

        # Check Generator Directory
        if not os.path.exists(GENERATOR_PATH):
            print('[-] Generation prompts directory Not Found')
            return

        #----------------------------------------------------------------
        # Load CSV
        #----------------------------------------------------------------


        print("###-------------------------------------###")
        print("### Loading Data")
        print("###-------------------------------------###\n\n")
        generator_df = pd.read_csv(os.path.join(GENERATOR_PATH, "prompts.csv"))

        # Read predictions
        if not os.path.isfile(generator_predict_file):
            print("#--ERROR--# Generator prediction file not found: " + generator_predict_file)
            raise ValueError("Generator prediction file not found: " + generator_predict_file)

        # Read the solution and prediction values into list
        self.selected_IDs = []
        self.generator_predictions = []
        with open(generator_predict_file, 'r') as f:
            raw_generator_predictions = f.read().split('\n\n\n\n')[:-1]
            for i, prediction in enumerate(raw_generator_predictions):
                if prediction.startswith("ID: "):
                    self.selected_IDs.append(int(raw_generator_predictions[i].split('\n')[0].split('ID: ')[1]))
                    self.generator_predictions.append('\n'.join(raw_generator_predictions[i].split('\n')[1:]))
                else:
                    self.selected_IDs.append(i)
                    self.generator_predictions.append(prediction)



        # match the order generator_df['id'] with self.selected_IDs
        generator_df = generator_df.set_index('id')
        generator_df = generator_df.loc[self.selected_IDs]
        generator_df = generator_df.reset_index()


        self.generator_prompts = generator_df['prompt'].values

        print("Loading generator solutions (good and bad papers)")
        self.generator_solutions = []
        for paper_id in generator_df['id'].values:
            if USE_OUR_BASELINE_REVIEWER:
                paraphrased_papers = {'good': [], 'bad': {}, 'human': []}
            else:
                paraphrased_papers = {'good': [], 'bad': {}}

            for paper_filename in tqdm([x for x in os.listdir(os.path.join(GENERATOR_PATH, "papers", str(paper_id))) if x!= ".DS_Store"]):
                if "soundness" in paper_filename:
                    continue
                
                if 'human' in paper_filename and not USE_OUR_BASELINE_REVIEWER:
                    continue

                paper_text = open(os.path.join(GENERATOR_PATH, "papers", str(paper_id), paper_filename), 'r').read()

                if paper_filename.startswith("human") and USE_OUR_BASELINE_REVIEWER:
                    paper_text = custom_json_loads(paper_text)
                    paper_text_reformat = [{'heading': 'Title', 'text': paper_text['title']}]
                    for section in paper_text['sections']:
                        paper_text_reformat.append({'heading': section['heading'], 'text': section['text']})
                    paper_references_reformat = []
                    for reference in paper_text['references']:
                        paper_references_reformat.append("@article{article1,\n title={"+reference['title']+"}\n}")
                    paper_text_reformat.append({'heading': 'References', 'text': "\n\n".join(paper_references_reformat)})
                    paper_text_reformat = json.dumps(paper_text_reformat)
                    paper_text_reformat = self.truncate_paper(paper_text_reformat)
                    paraphrased_papers['human'].append(paper_text_reformat)

                    continue

                paper_text = open(os.path.join(GENERATOR_PATH, "papers", str(paper_id), paper_filename), 'r').read()
                paper_text = self.truncate_paper(paper_text)

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
        

        self.truncate_generator_predictions()
        if (len(self.generator_solutions) != len(self.generator_predictions)): 
            print("#--ERROR--# Number of lines in solution file (" + str(len(self.generator_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.generator_predictions)) + ")")
            raise ValueError("Number of lines in solution file (" + str(len(self.generator_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.generator_predictions)) + ")")

        print("###-------------------------------------###")
        print("### Solutions files are ready!")
        print("###-------------------------------------###\n\n")
        

    def get_word_count(self, prediction):
        word_count = 0
        if isinstance(prediction, str):
            print("WARNING: prediction is a string. Converting to json.")
            prediction_json = custom_json_loads(prediction)
        else:
            prediction_json = prediction
        for section in prediction_json:
            if section['heading'] != 'References':
                word_count += len(word_tokenize(section['heading'])) + len(word_tokenize(section['text']))
        return word_count

    def truncate_paper(self, paper, max_length=2000, return_as_json=False):
        if isinstance(paper, str):
            paper_json = custom_json_loads(paper)
        else:
            paper_json = paper
        word_count = self.get_word_count(paper_json)
        if word_count > max_length:
            while word_count > max_length:
                truncated_paper_json = []
                for section in paper_json:
                    if section['heading'] == 'References':
                        truncated_paper_json.append(section)
                    else:
                        truncated_paper_json.append({'heading': section['heading'], 'text': ' '.join(sent_tokenize(section['text'])[:-1])})
                paper_json = truncated_paper_json
                word_count = self.get_word_count(paper_json)
        if return_as_json:
            return paper_json
        else:
            return json.dumps(paper_json)

    def truncate_reference(self, paper, soft_max_num_references=10, hard_max_num_references=15, return_as_json=False):
        if isinstance(paper, str):
            paper_json = custom_json_loads(paper)
        else:
            paper_json = paper
        
        reference_idx = None
        reference_count = 0
        reference_list = []
        for i, section in enumerate(paper_json):
            if section['heading'] == 'References':
                reference_idx = i
                for reference in section['text'].split('\n\n'):
                    if reference != '':
                        reference_count += 1
                        reference_list.append(reference)
        if reference_count > hard_max_num_references:
            print("ERROR: reference count is more than " + str(hard_max_num_references) + ". Exiting.")
            raise ValueError("Reference count is more than " + str(hard_max_num_references) + ".")
        if reference_count > soft_max_num_references:
            print("WARNING: reference count is more than " + str(soft_max_num_references) + ". Truncating.")
            truncated_reference_list = reference_list[:soft_max_num_references]
            paper_json[reference_idx]['text'] = '\n\n'.join(truncated_reference_list)

        if return_as_json:
            return paper_json
        else:
            return json.dumps(paper_json)
            

    def truncate_generator_predictions(self, soft_max_length=2000, hard_max_length=2500):
        truncate_generator_predictions = []
        for prediction in self.generator_predictions:
            if isinstance(prediction, str):
                prediction_json = custom_json_loads(prediction)
            else:
                prediction_json = prediction
            word_count = self.get_word_count(prediction_json)


            if word_count > hard_max_length:
                print("ERROR: prediction is longer than " + str(hard_max_length) + " tokens. Exiting.")
                raise ValueError("Prediction is longer than " + str(hard_max_length) + " tokens.") 
            
            if word_count > soft_max_length:
                print("WARNING: prediction is longer than " + str(soft_max_length) + " tokens. Truncating.")
                prediction_json = self.truncate_paper(prediction_json, max_length=soft_max_length, return_as_json=True)
                print(f"The total word count of the paper (excluding references) is now: {word_count}")
            
            prediction_json = self.truncate_reference(prediction_json, return_as_json=True)
            
            truncate_generator_predictions.append(json.dumps(prediction_json))
        self.generator_predictions = truncate_generator_predictions


    def get_generator_scores(self):
        '''Function to get the overall score for generator'''
        if self.overall_generator_score is None:
            self.compute_generator_scores()
        return self.overall_generator_score, self.generator_scores, self.generator_html_comments

    def compute_generator_scores(self):
        '''Function to compute the score for generator'''

        # Compute the score for each criterion
        self.generator_scores = []
        print("Start computing scores for each generated paper")
        for i in tqdm(range(len(self.generator_solutions))):
            generator_score = {}
            prediction_prompt = self.generator_prompts[i]
            prediction_paper = self.generator_predictions[i]
            solution_papers = self.generator_solutions[i]
            good_papers = solution_papers['good']
            bad_papers = solution_papers['bad'] 
            if USE_OUR_BASELINE_REVIEWER:
                human_papers = solution_papers['human'] 

            available_criteria = defaultdict(dict)
            for super_category, super_value in solution_papers['bad'].items():
                if isinstance(super_value, list):
                    available_criteria[super_category] = "0 if paper 1 is better, 1 if paper 2 is better"
                else:
                    for sub_category in super_value:
                        available_criteria[super_category][sub_category] = "0 if paper 1 is better, 1 if paper 2 is better"

            # print("\nGenerating comments for each generated paper") 
            if DEBUG:
                generator_score = {'relevance': {'title': 0.583, 'abstract': 0.75, 'citations': 0.166}, 'contribution': {'conclusion': 0.583, 'abstract': 0.75, 'coverage': 0.166}, 'responsibility': 0.166, 'clarity': {'explanations': 0.66, 'correctlanguage': 0.16, 'organization': 0.16666666666666666}, 'soundness': {'c1': 0.9, 'c2': 0.8, 'c3': 0.9}}
                html_comment = {'relevance': {'title': "TEST", 'abstract': "TEST", 'citations': "TEST"}, 'contribution': {'conclusion': "TEST", 'abstract': "TEST", 'coverage': "TEST"}, 'responsibility': "TEST", 'clarity': {'explanations': "TEST", 'correctlanguage': "TEST", 'organization': "TEST"}, 'soundness': {'c1': "TEST", 'c2': "TEST", 'c3': "TEST"}}
            else:
                if USE_OUR_BASELINE_REVIEWER:
                    answer = self.baseline_reviewer.compare_papers(good_papers, bad_papers, prediction_paper, prediction_prompt, available_criteria, human_papers)
                else:
                    answer = self.baseline_reviewer.compare_papers(good_papers, bad_papers, prediction_paper, prediction_prompt, available_criteria)
                generator_score = answer[0]
                html_comment = answer[1]
             
            self.generator_scores.append(generator_score)
            self.generator_html_comments.append(html_comment)

            print("\nGenerated score for paper " + str(i))

        # Combine the scores accross all papers
        self.overall_generator_score = defaultdict(lambda: defaultdict(int))
        # print(self.generator_scores)
        for generator_score in self.generator_scores:
            for super_category, super_value in generator_score.items():
                if isinstance(super_value, dict):
                    for sub_category, sub_value in super_value.items():
                        self.overall_generator_score[super_category][sub_category] += float(sub_value)/len(self.generator_scores)
                else:
                    if super_category not in self.overall_generator_score:
                        self.overall_generator_score[super_category] = 0
                    self.overall_generator_score[super_category] += float(super_value)/len(self.generator_scores)       

    def get_overall_generator_scores(self):
        """Get the overall scores."""
        if self.overall_generator_score is None:
            self.compute_generator_scores()

        generator_score_no_relevance = 0
        generator_relevance_score = 0
        num_criteria_without_relevance = len(self.overall_generator_score)
        for key, super_value in self.overall_generator_score.items():
            if key == "relevance":
                num_criteria_without_relevance -= 1
                for _, sub_value in super_value.items():
                    generator_relevance_score += sub_value / len(super_value)
                continue
            if isinstance(super_value, dict):
                for _, sub_value in super_value.items():
                    generator_score_no_relevance += sub_value / ( len(super_value) * num_criteria_without_relevance )
            else:
                generator_score_no_relevance += super_value / num_criteria_without_relevance
        generator_score = generator_score_no_relevance * generator_relevance_score
        return generator_score
    
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
            html_file.write("<ul>\n")
            if isinstance(super_value, dict):
                values = list(super_value.values())
                average = np.mean(values)
                color = 'red' if average < 0.5 else 'black'

                if CONTESTANT_MODE:
                    reasons = []
                    if types is None:
                        html_file.write(f"<li><b>{super_category}</b>: <span style='color:{color};'>{average:.2f}</span></li>\n")
                    else:
                        for sub_category, sub_value in super_value.items():
                            reasons.append(str(html_comments[super_category][sub_category]))
                        reason = " || ".join(reasons)

                        if not DEBUG:
                            conversation = [{"role": "system", "content": "You are a helpful assistant who will help me combine all reviews into one. Focus on the text and not the individual score."},
                                            {"role": "user", "content": reason}]
                            reason = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]

                            
                            
                        html_file.write(f"<li><b>{super_category}</b>: <span style='color:{color};'>{average:.2f}</span><br>&emsp;reason: {reason}</li>\n")
                else:
                    html_file.write(f"<li><b>{super_category}</b>: <span style='color:{color};'>{average:.2f}</span></li>\n")
                    for sub_category, sub_value in super_value.items():
                        html_file.write("<ul>\n")

                        color = 'red' if sub_value < 0.5 else 'black'
                        if types is None:
                            html_file.write(f"<li>{sub_category}: <span style='color:{color};'>{sub_value:.2f}</span></li>\n")
                        else:
                            reason = "" if html_comments == "" else html_comments[super_category][sub_category]
                            html_file.write(f"<li>{sub_category}: <span style='color:{color};'>{sub_value:.2f}</span><br>&emsp;reason: {reason}</li>\n")
                        html_file.write("</ul>\n")
            
            else:
                color = 'red' if super_value < 0.5 else 'black'
                if types is None:
                    html_file.write(f"<li><b>{super_category}</b>: <span style='color:{color};'>{super_value:.2f}</span></li>\n")
                else:
                    reason = "" if html_comments == "" else html_comments[super_category]
                    html_file.write(f"<li><b>{super_category}</b>: <span style='color:{color};'>{super_value:.2f}</span></li>&emsp;reason: {reason}\n")
            html_file.write("</ul>\n")
