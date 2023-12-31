import os
from os.path import isfile
import numpy as np
import pandas as pd
import json5 as json
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import base64
from scipy import stats
import re
from tqdm.auto import tqdm

from collections import defaultdict
from libscores import kendall_tau, safe_kendalltau

from metacriteria.utils import custom_json_loads

from meta_text_reviewer import MetaTextReviewer

LIMIT_COMMENT_LENGTH = 2000

class Evaluator:
    def __init__(self):
        self.EVALUATION_MODE = None
        self.solution_dir = None
        self.super_categories = ['relevance', 'clarity', 'contribution', 'soundness', 'responsibility', 'overall'] #, 'confidence'

        self.super_categories_for_text_reviewer = ['relevance', 'clarity', 'contribution', 'soundness', 'responsibility']
        # Reviewer
        self.reviewer_predictions = None
        self.reviewer_solutions = None
        self.reviewer_metadata = None
        self.numeric_reviewer_ranking_scores = None
        self.overall_numeric_reviewer_scores = None
        self.text_reviewer_scores = None
        self.overall_text_reviewer_scores = None
        self.overall_reviewer_scores = None

        self.average_score_of_good_papers_text_meta_review = None
        self.average_score_of_bad_papers_text_meta_review = None
        self.pairs_of_good_and_bad_scores = None

        self.three_highest_score_paper_details = None
        self.three_lowest_score_paper_details = None

        self.meta_text_reviewer = MetaTextReviewer()

    def read_reviewer_solutions_and_predictions(self, solution_dir, reviewer_predict_file):
        ''' Function to read the Labels from CSV files'''

        self.solution_dir = solution_dir

        print("###-------------------------------------###")
        print("### Checking Data")
        print("###-------------------------------------###\n\n")

        #----------------------------------------------------------------
        # Settings
        #----------------------------------------------------------------
        REVIEWER_PATH = os.path.join(solution_dir, "reviewer")


        #----------------------------------------------------------------
        # Errors
        #----------------------------------------------------------------

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
        self.reviewer_metadata = pd.read_csv(os.path.join(REVIEWER_PATH, "metadata.csv"))
        
        # Reviewer
        self.reviewer_solutions = self.reviewer_metadata[['id', 'pdf_name', 'good_or_bad']]
        # for i in range(len(reviewer_df)):
        #     reviewer_solution = {}
        #     for criterion in self.reviewer_criteria:
        #         reviewer_solution[criterion] = reviewer_df[criterion][i]
        #     self.reviewer_solutions.append(reviewer_solution)

        if not os.path.isfile(reviewer_predict_file):
            print("#--ERROR--# Reviewer prediction file not found: " + reviewer_predict_file)
            raise ValueError("Reviewer prediction file not found: " + reviewer_predict_file)

        # Read the solution and prediction values into list
        selected_IDs = []
        self.reviewer_predictions = {}
        with open(reviewer_predict_file, 'r') as f:
            full_reviewer_predictions = f.read()
            raw_reviewer_predictions = full_reviewer_predictions.split('\n\n\n\n')[:-1]
            for i, prediction in enumerate(raw_reviewer_predictions):
                if prediction.startswith("ID: "):
                    selected_ID = int(raw_reviewer_predictions[i].split('\n')[0].split('ID: ')[1])
                    selected_IDs.append(selected_ID)
                    self.reviewer_predictions[selected_ID] = '\n'.join(raw_reviewer_predictions[i].split('\n')[1:])
                else:
                    selected_ID = i
                    selected_IDs.append(selected_ID)
                    self.reviewer_predictions[selected_ID] = prediction

            self.EVALUATION_MODE = full_reviewer_predictions.split('\n\n\n\n')[-1].split('EVALUATION_MODE: ')[1]
            

        # use double quotes instead of single quotes
        loaded_reviewer_predictions = {}
        for key, x in self.reviewer_predictions.items():
            try:
                loaded_reviewer_predictions[key] = custom_json_loads(x)
            except:
                print("#--ERROR--# Reviewer prediction file is not in JSON format: " + x)
                # Add a comma after every "}" if THE COMMA IS NOT THERE
                x = re.sub(r'}(?!\s*,)', r'},', x)
                while not x.endswith("}"):
                    x = x[:-1]
                loaded_reviewer_predictions[key] = custom_json_loads(x)
        self.reviewer_predictions = loaded_reviewer_predictions

        # remove all spaces from the keys and keys of values of each of reviewer prediction
        for super_key, super_value in self.reviewer_predictions.items():
            new_prediction = {}
            for key, value in super_value.items():
                new_key = key.lower().replace(" ", "")
                if len(value['comment']) >= LIMIT_COMMENT_LENGTH:
                    raise ValueError("Reviewer prediction comment is too long: " + value['comment'])
                new_prediction[new_key] = value
            self.reviewer_predictions[super_key] = new_prediction

        # select only the solutions that are in the selected IDs, matching the order of the selected IDs
        self.reviewer_solutions = self.reviewer_solutions.set_index('id')
        self.reviewer_solutions = self.reviewer_solutions.loc[selected_IDs]
        self.reviewer_solutions = self.reviewer_solutions.reset_index()

        # Check if the number of lines in the solution file matches the number of lines in the prediction file
        if (len(self.reviewer_solutions) != len(self.reviewer_predictions)):
            print("#--ERROR--# Number of lines in solution file (" + str(len(self.reviewer_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.reviewer_predictions)) + ")")
            raise ValueError("Number of lines in solution file (" + str(len(self.reviewer_solutions)) + ") does not match number of lines in prediction file (" + str(len(self.reviewer_predictions)) + ")")
        
        print("###-------------------------------------###")
        print("### Solutions files are ready!")
        print("###-------------------------------------###\n\n")
        
    def get_number_of_papers(self):
        if self.reviewer_solutions is None:
            print("#--ERROR--# Reviewer solutions are not loaded")
            raise ValueError("Reviewer solutions are not loaded")
        return len(self.reviewer_solutions)

    def get_generic_reviewer_scores(self, code):
        self.meta_text_reviewer.set_api_key(code)
        if code == "key1":
            return self.get_numeric_reviewer_scores()
        elif code == "key2":
            return self.get_good_paper_text_reviewer_scores()
        elif code == "key3":
            return self.get_bad_paper_text_reviewer_scores()
        
    def get_numeric_reviewer_scores(self):
        """Get the reviewer scores for each criterion"""
        if self.overall_numeric_reviewer_scores is None:
            self.compute_reviewer_scores()
        return self.overall_numeric_reviewer_scores 

    def get_good_paper_text_reviewer_scores(self):
        """Get the text reviewer scores for each criterion for good papers"""
        if self.average_score_of_good_papers_text_meta_review is None:
            self.compute_reviewer_scores()
        return self.average_score_of_good_papers_text_meta_review

    def get_bad_paper_text_reviewer_scores(self):
        """Get the text reviewer scores for each criterion for bad papers"""
        if self.average_score_of_bad_papers_text_meta_review is None:
            self.compute_reviewer_scores()
        return self.average_score_of_bad_papers_text_meta_review

    def get_three_highest_score_paper_details(self):
        """Get the three highest score paper details"""
        if self.three_highest_score_paper_details is None:
            self.compute_reviewer_scores()
        return self.three_highest_score_paper_details

    def get_three_lowest_score_paper_details(self):
        """Get the three lowest score paper details"""
        if self.three_lowest_score_paper_details is None:
            self.compute_reviewer_scores()
        return self.three_lowest_score_paper_details

    def compute_reviewer_scores(self):
        """Compute the reviewer scores."""
        print("(Contrastive Eval) Evaluating how well you can differentiate good papers from bad papers...")
        self.numeric_reviewer_ranking_scores = {}
        
        for solution_type in self.reviewer_solutions['good_or_bad'].unique():
            if 'good' in solution_type or 'human' in solution_type:
                continue
            criterion = solution_type.split("_")[1]
            for super_category in self.super_categories:
                if criterion.startswith(super_category):
                    self.numeric_reviewer_ranking_scores[super_category] = []
                    break

        # loop over all the criteria to get the scores
        good_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: 'good_1' in x)]
        good_scores_ids_as_index = good_scores_df.groupby('pdf_name')['id'].apply(list)
        good_scores_ids = good_scores_ids_as_index.values

        bad_scores_ids_dict = {}
        for criterion in self.numeric_reviewer_ranking_scores:
            bad_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: criterion in x)]
            bad_scores_ids_as_index = bad_scores_df.groupby('pdf_name')['id'].apply(list)
            bad_scores_ids_as_index = bad_scores_ids_as_index[good_scores_ids_as_index.index]
            bad_scores_ids = bad_scores_ids_as_index.values

            bad_scores_ids_dict[criterion] = bad_scores_ids

        # ============== NUMERIC REVIEWER RANKING SCORES ==============
        # Calculate ranking score of: RESPONSIILITY, SOUNDNESS, CONTRIBUTION, CLARITY
        # loop over each set of papers
        self.pairs_of_good_and_bad_scores = {criterion: [] for criterion in self.numeric_reviewer_ranking_scores}
        average_ranking_score_across_all_sets = []
        for set_i in range(len(good_scores_ids)):
            average_ranking_score_each_set = []
            # loop over each criterion
            for criterion in self.numeric_reviewer_ranking_scores:
                # loop over each paper in the good set
                for good_j in range(len(good_scores_ids[set_i])):
                    good_score = self.reviewer_predictions[good_scores_ids[set_i][good_j]][criterion]["score"]
                    all_bad_scores = []
                    # loop over all paper in the bad set for that criterion
                    for bad_j in range(len(bad_scores_ids_dict[criterion][set_i])):
                        bad_score = self.reviewer_predictions[bad_scores_ids_dict[criterion][set_i][bad_j]][criterion]["score"]
                        all_bad_scores.append(bad_score)
                    
                    # compute the ranking of the good score among all the bad scores, with 0 being the lowest and 1 being the highest
                    rank = stats.percentileofscore(all_bad_scores, good_score, kind='strict') / 100

                    self.numeric_reviewer_ranking_scores[criterion].append(rank)
                    average_ranking_score_each_set.append(rank)

                    # save the good and bad scores for each criterion
                    for bad_score in all_bad_scores:
                        self.pairs_of_good_and_bad_scores[criterion].append((good_score, bad_score))
            
            average_ranking_score_across_all_sets.append(np.mean(average_ranking_score_each_set))
            
        
        # # Calculate ranking score of: CONFIDENCE
        
        # confidence_score_each_set = []
        # for set_i in range(len(good_scores_ids)):
        #     confidence_score_each_set.append(self.reviewer_predictions[good_scores_ids[set_i][0]]["confidence"]["score"])
        
        # print("Average ranking score across all sets:", average_ranking_score_across_all_sets)
        # print("Predicted confidence score each set:", confidence_score_each_set)

        # # compute the confidence score evaluation, with 0 being the lowest and 1 being the highest
        # # the evaluation should penalize the confidence score if the average score is low while the confidence score is high, and vice versa
        # # one way to do this is to compute the correlation between the average score and the confidence score
        # self.numeric_reviewer_ranking_scores["confidence"] = []
        # for set_i in range(len(good_scores_ids)):
        #     # correlation = cov(x, y) / (std(x) * std(y))
        #     correlation = stats.pearsonr(average_ranking_score_across_all_sets, confidence_score_each_set)[0]
        #     # scale the correlation to be between 0 and 1
        #     correlation = (correlation + 1) / 2
        #     self.numeric_reviewer_ranking_scores["confidence"].append(0 if np.isnan(correlation) else correlation)

        # print("###-------------------------------------###")
        # print("### Detailed Numeric Reviewer Ranking Scores")
        # print(self.numeric_reviewer_ranking_scores)

        # Calculate the overall reviewer scores for each criterion
        self.overall_numeric_reviewer_scores = {}
        contrastive_eval_different_than_0 = False
        for criterion in self.numeric_reviewer_ranking_scores:
            self.overall_numeric_reviewer_scores[criterion] = np.mean(self.numeric_reviewer_ranking_scores[criterion])
            if self.overall_numeric_reviewer_scores[criterion] != 0:
                contrastive_eval_different_than_0 = True

        if self.EVALUATION_MODE == 'full':

            # ============== TEXT REVIEWER RANKING SCORES ==============

            # Get the generated scores and comments for each set of papers
            all_good_scores_and_comments = []
            for set_i in range(len(good_scores_ids)):
                all_good_scores_and_comments.append([self.reviewer_predictions[good_scores_ids[set_i][0]]])

            all_bad_scores_and_comments = {}
            bad_scores_df = self.reviewer_solutions[self.reviewer_solutions['good_or_bad'].apply(lambda x: 'bad' in x)]
            bad_scores_ids_as_index = bad_scores_df.groupby('pdf_name')['id'].apply(list)
            bad_scores_ids_as_index = bad_scores_ids_as_index[good_scores_ids_as_index.index]
            bad_scores_ids = bad_scores_ids_as_index.values
            for set_i in range(len(good_scores_ids)):
                all_bad_scores_and_comments[set_i] = []
                for bad_j in range(len(bad_scores_ids[set_i])):
                    all_bad_scores_and_comments[set_i].append(self.reviewer_predictions[bad_scores_ids[set_i][bad_j]])

            # Use the meta text reviewer to get the scores of each set of papers, for each criterion
            # Good paper set
            print("(LLM Eval) Evaluating good paper set...")
            all_meta_review_of_good_scores_and_comments = []
            for set_i in tqdm(range(len(good_scores_ids))):
                good_scores_and_comments = all_good_scores_and_comments[set_i]

                # Get the meta-review of the good scores and comments
                # This will return a list of dictionary, each value in the dictionary coresponds with a list of 5 scores for each criterion:
                # - A - Score: Is the score consistent with the text feed-back?
                # - B - Precision (clarity): Is the text feed-back precise (does it point to a specific reason of praise of criticism)?
                # - C - Correctness (soundness): Is the praise or criticism correct and well substantiated?
                # - D - Recommendation (contribution): Does the text feed-back provide detailed and actionable recommendations for improvement?
                # - E - Respectfulness (responsibility): Is the language polite and non discriminatory?

                if contrastive_eval_different_than_0:
                    meta_review_of_good_scores_and_comments = self.meta_text_reviewer.get_meta_review_scores(good_scores_and_comments, self.super_categories_for_text_reviewer)
                else:
                    # Return all 0 scores
                    meta_review_of_good_scores_and_comments = [{criterion: np.zeros(5) for criterion in self.super_categories_for_text_reviewer}]
                all_meta_review_of_good_scores_and_comments.append(meta_review_of_good_scores_and_comments)
            # print("all_meta_review_of_good_scores_and_comments:", all_meta_review_of_good_scores_and_comments)
            # Bad paper set
            print("(LLM eval) Evaluating bad paper set...")
            all_meta_review_of_bad_scores_and_comments = []
            for set_i in tqdm(range(len(good_scores_ids))):
                bad_scores_and_comments = all_bad_scores_and_comments[set_i]

                # Get the meta-review of the bad scores and comments
                # This will return a list of 5 scores for each criterion:
                # - A - Score: Is the score consistent with the text feed-back?
                # - B - Precision (clarity): Is the text feed-back precise (does it point to a specific reason of praise of criticism)?
                # - C - Correctness (soundness): Is the praise or criticism correct and well substantiated?
                # - D - Recommendation (contribution): Does the text feed-back provide detailed and actionable recommendations for improvement?
                # - E - Respectfulness (responsibility): Is the language polite and non discriminatory?

                if contrastive_eval_different_than_0:
                    meta_review_of_bad_scores_and_comments = self.meta_text_reviewer.get_meta_review_scores(bad_scores_and_comments, self.super_categories_for_text_reviewer)
                else:
                    # Return all 0 scores
                    meta_review_of_bad_scores_and_comments = [{criterion: np.zeros(5) for criterion in self.super_categories_for_text_reviewer}]
                all_meta_review_of_bad_scores_and_comments.append(meta_review_of_bad_scores_and_comments)

            # Calculate the average score of each meta-criterion
            three_lowest_scores = [np.inf, np.inf, np.inf]
            three_highest_scores = [-np.inf, -np.inf, -np.inf]

            three_lowest_score_ids = [None, None, None]
            three_highest_score_ids = [None, None, None]
            
            self.three_lowest_score_paper_details = [None, None, None]
            self.three_highest_score_paper_details = [None, None, None]
            # Good paper set
            debug_all_text_meta_review_score_of_each_paper = []
            self.average_score_of_good_papers_text_meta_review = np.zeros((len(self.super_categories_for_text_reviewer),5))
            for set_i in range(len(good_scores_ids)):
                meta_review_of_good_scores_and_comments = all_meta_review_of_good_scores_and_comments[set_i]
                for paper_j, meta_review_of_each_paper in enumerate(meta_review_of_good_scores_and_comments):
                    text_meta_review_score_of_each_paper = np.zeros((len(self.super_categories_for_text_reviewer),5))
                    for crit_j, criterion in enumerate(self.super_categories_for_text_reviewer):
                        text_meta_review_score_of_each_paper[crit_j] += meta_review_of_each_paper[criterion]
                    self.average_score_of_good_papers_text_meta_review += text_meta_review_score_of_each_paper
                    debug_all_text_meta_review_score_of_each_paper.append(text_meta_review_score_of_each_paper)
                    average_text_meta_review_score_of_each_paper = np.mean(text_meta_review_score_of_each_paper)

                    #Update the lowest score and the lowest score paper
                    if average_text_meta_review_score_of_each_paper < three_lowest_scores[2]:
                        # Get the paper id of the lowest score
                        lowest_score_id = good_scores_ids[set_i][paper_j]
                        # Get the predicted review of the lowest score
                        lowest_score_review = self.reviewer_predictions[lowest_score_id]
                        # # Get the meta-review reason of the lowest score
                        # lowest_score_meta_review_reason = self.meta_text_reviewer.get_meta_review_reasons(lowest_score_review, meta_review_of_each_paper, self.super_categories_for_text_reviewer)

                        current_position = 1
                        while current_position >= 0 and average_text_meta_review_score_of_each_paper < three_lowest_scores[current_position]:
                            three_lowest_scores[current_position] = three_lowest_scores[current_position - 1]
                            self.three_lowest_score_paper_details[current_position] = self.three_lowest_score_paper_details[current_position - 1]
                            current_position -= 1
                        three_lowest_scores[current_position + 1] = average_text_meta_review_score_of_each_paper
                        three_lowest_score_ids[current_position + 1] = lowest_score_id
                        self.three_lowest_score_paper_details[current_position + 1] = {
                            "id": lowest_score_id,
                            "review": lowest_score_review,
                            "meta_review_score": meta_review_of_each_paper,
                            # "meta_review_reason": lowest_score_meta_review_reason
                        }
                    
                    # Update the highest score and the highest score paper
                    if average_text_meta_review_score_of_each_paper > three_highest_scores[2]:

                        # Get the paper id of the highest score
                        highest_score_id = good_scores_ids[set_i][paper_j]
                        # Get the predicted review of the highest score
                        highest_score_review = self.reviewer_predictions[highest_score_id]
                        # # Get the meta-review reason of the highest score
                        # highest_score_meta_review_reason = self.meta_text_reviewer.get_meta_review_reasons(highest_score_review, meta_review_of_each_paper, self.super_categories_for_text_reviewer)

                        current_position = 1
                        while current_position >= 0 and average_text_meta_review_score_of_each_paper > three_highest_scores[current_position]:
                            three_highest_scores[current_position] = three_highest_scores[current_position - 1]
                            self.three_highest_score_paper_details[current_position] = self.three_highest_score_paper_details[current_position - 1]
                            current_position -= 1
                        three_highest_scores[current_position + 1] = average_text_meta_review_score_of_each_paper
                        three_highest_score_ids[current_position + 1] = highest_score_id
                        self.three_highest_score_paper_details[current_position + 1] = {
                            "id": highest_score_id,
                            "review": highest_score_review,
                            "meta_review_score": meta_review_of_each_paper,
                            # "meta_review_reason": highest_score_meta_review_reason
                        }

            self.average_score_of_good_papers_text_meta_review = self.average_score_of_good_papers_text_meta_review/(len(good_scores_ids) * len(meta_review_of_good_scores_and_comments))

            # Bad paper set
            self.average_score_of_bad_papers_text_meta_review = np.zeros((len(self.super_categories_for_text_reviewer),5))
            for set_i in range(len(good_scores_ids)):
                meta_review_of_bad_scores_and_comments = all_meta_review_of_bad_scores_and_comments[set_i]
                for meta_review_of_each_paper in meta_review_of_bad_scores_and_comments:
                    text_meta_review_score_of_each_paper = np.zeros((len(self.super_categories_for_text_reviewer),5))
                    for crit_j, criterion in enumerate(self.super_categories_for_text_reviewer):
                        text_meta_review_score_of_each_paper[crit_j] += meta_review_of_each_paper[criterion]
                    self.average_score_of_bad_papers_text_meta_review += text_meta_review_score_of_each_paper
                    debug_all_text_meta_review_score_of_each_paper.append(text_meta_review_score_of_each_paper)
                    average_text_meta_review_score_of_each_paper = np.mean(text_meta_review_score_of_each_paper)

                    # Update the lowest score and the lowest score paper
                    if average_text_meta_review_score_of_each_paper < three_lowest_scores[2]:
                        # Get the paper id of the lowest score
                        lowest_score_id = bad_scores_ids[set_i][paper_j]
                        # Get the predicted review of the lowest score
                        lowest_score_review = self.reviewer_predictions[lowest_score_id]
                        # # Get the meta-review reason of the lowest score
                        # lowest_score_meta_review_reason = self.meta_text_reviewer.get_meta_review_reasons(lowest_score_review, meta_review_of_each_paper, self.super_categories_for_text_reviewer)

                        current_position = 1
                        while current_position >= 0 and average_text_meta_review_score_of_each_paper < three_lowest_scores[current_position]:
                            three_lowest_scores[current_position] = three_lowest_scores[current_position - 1]
                            self.three_lowest_score_paper_details[current_position] = self.three_lowest_score_paper_details[current_position - 1]
                            current_position -= 1
                        three_lowest_scores[current_position + 1] = average_text_meta_review_score_of_each_paper
                        three_lowest_score_ids[current_position + 1] = lowest_score_id
                        self.three_lowest_score_paper_details[current_position + 1] = {
                            "id": lowest_score_id,
                            "review": lowest_score_review,
                            "meta_review_score": meta_review_of_each_paper,
                            # "meta_review_reason": lowest_score_meta_review_reason
                        }


                    # Update the highest score and the highest score paper
                    if average_text_meta_review_score_of_each_paper > three_highest_scores[2]:
                        # Get the paper id of the highest score
                        highest_score_id = bad_scores_ids[set_i][paper_j]
                        # Get the predicted review of the highest score
                        highest_score_review = self.reviewer_predictions[highest_score_id]
                        # # Get the meta-review reason of the highest score
                        # highest_score_meta_review_reason = self.meta_text_reviewer.get_meta_review_reasons(highest_score_review, meta_review_of_each_paper, self.super_categories_for_text_reviewer)

                        current_position = 1
                        while current_position >= 0 and average_text_meta_review_score_of_each_paper > three_highest_scores[current_position]:
                            three_highest_scores[current_position] = three_highest_scores[current_position - 1]
                            self.three_highest_score_paper_details[current_position] = self.three_highest_score_paper_details[current_position - 1]
                            current_position -= 1
                        three_highest_scores[current_position + 1] = average_text_meta_review_score_of_each_paper
                        three_highest_score_ids[current_position + 1] = highest_score_id
                        self.three_highest_score_paper_details[current_position + 1] = {
                            "id": highest_score_id,
                            "review": highest_score_review,
                            "meta_review_score": meta_review_of_each_paper,
                            # "meta_review_reason": highest_score_meta_review_reason
                        }

            # Now calculate the details of the three highest and lowest score papers
            for i in range(3):
                mean_highest_score = 0
                mean_lowest_score = 0
                for criterion in self.super_categories_for_text_reviewer:
                    mean_highest_score += np.mean(self.three_highest_score_paper_details[i]["meta_review_score"][criterion])
                    mean_lowest_score += np.mean(self.three_lowest_score_paper_details[i]["meta_review_score"][criterion])
                if mean_highest_score == 0 and mean_lowest_score == 0:
                    self.three_highest_score_paper_details[i]["meta_review_reason"] = {criterion: ["No reason"]*5 for criterion in self.super_categories_for_text_reviewer}
                    self.three_lowest_score_paper_details[i]["meta_review_reason"] = {criterion: ["No reason"]*5 for criterion in self.super_categories_for_text_reviewer}
                else:
                    self.three_highest_score_paper_details[i]["meta_review_reason"] = self.meta_text_reviewer.get_meta_review_reasons(self.three_highest_score_paper_details[i]["review"], self.three_highest_score_paper_details[i]["meta_review_score"], self.super_categories_for_text_reviewer)
                    self.three_lowest_score_paper_details[i]["meta_review_reason"] = self.meta_text_reviewer.get_meta_review_reasons(self.three_lowest_score_paper_details[i]["review"], self.three_lowest_score_paper_details[i]["meta_review_score"], self.super_categories_for_text_reviewer)
            
            self.average_score_of_bad_papers_text_meta_review = self.average_score_of_bad_papers_text_meta_review/(len(good_scores_ids) * len(meta_review_of_bad_scores_and_comments))
            
            # print("###-------------------------------------###")
            # print("### Detailed Text Reviewer Ranking Scores")
            # print("###-------------------------------------###")
            # print(debug_all_text_meta_review_score_of_each_paper)

        elif self.EVALUATION_MODE == 'fast':
            # Return 0 scores
            self.average_score_of_good_papers_text_meta_review = np.zeros((len(self.super_categories_for_text_reviewer),5))
            self.average_score_of_bad_papers_text_meta_review = np.zeros((len(self.super_categories_for_text_reviewer),5))
            self.three_lowest_score_paper_details = [{"id": None, "review": None, "meta_review_score": None, "meta_review_reason": None}]*3
            self.three_highest_score_paper_details = [{"id": None, "review": None, "meta_review_score": None, "meta_review_reason": None}]*3
        else:
            raise NotImplementedError("Evaluation mode not implemented")

        # Average the scores of text meta-reviewer
        self.overall_text_reviewer_scores = {}
        for i, criterion in enumerate(self.super_categories_for_text_reviewer):
            self.overall_text_reviewer_scores[criterion] = np.mean(self.average_score_of_good_papers_text_meta_review[i] + self.average_score_of_bad_papers_text_meta_review[i]) / 2


        # ============== (NUMERIC + TEXT) REVIEWER SCORES ==============
        self.overall_reviewer_scores = self.overall_numeric_reviewer_scores.copy()
        
        for criterion in self.overall_text_reviewer_scores:
            self.overall_reviewer_scores[criterion] = (self.overall_text_reviewer_scores[criterion] + self.overall_numeric_reviewer_scores[criterion])/2


    def get_overall_reviewer_scores(self):
        """Get the overall scores."""
        if self.overall_reviewer_scores is None:
            self.compute_reviewer_scores()

        reviewer_score = 0
        for criterion in self.overall_reviewer_scores:
            reviewer_score += self.overall_reviewer_scores[criterion]
        reviewer_score /= len(self.overall_reviewer_scores)
        
        return reviewer_score
    
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

    def write_numeric_json_score_to_html(self, json_score, html_file):
        """Convert the json score to text."""
        average_score = 0
        for criterion, score in json_score.items():
            # skip the confidence score
            if criterion == "confidence":
                continue
            average_score += score
            color = 'red' if score < 0.5 else 'black'
            html_file.write(f"&nbsp;{criterion}: <span style='color:{color};'>{score:.2f}</span><br>")
        # Write the overall score
        average_score /= len(json_score) - (1 if "confidence" in json_score else 0)
        color = 'red' if average_score < 0.5 else 'black'
        html_file.write(f"<br>&nbsp;Overall score: <span style='color:{color};'>{average_score:.2f}</span><br>")



    def plot_reviewer_scores_to_html(self, text_reviewer_scores, html_file):
        """Plot the reviewer scores as a 5x5 table to html file.
        Also put the name of each column on top of the table: Rating, Precision, Correctness, Recommendation, Respectfulness
        Also put the name of each row on the left of the table: Clarify, Contribution, Soundness, Responsibility
        Args:
            text_reviewer_scores: The text reviewer scores in a 5x5 numpy array.
            html_file: The html file to write the table to.
        """
        # Plot the table
        # sns.set(font_scale=2.0)
        plt.rc('font', size=12)
        fig = plt.figure()
        text_reviewer_scores_round_2_decimal = np.round(text_reviewer_scores, 2)
        # colLabels = Rating, Precision, Correctness, Recommendation, Respectfulness
        # rowLabels = Clarify, Contribution, Soundness, Responsibility
        # Plot heatmap using seaborn
        x_axis_labels = ["Rating", "Precision", "Correctness", "Recommendation", "Respectfulness"]
        y_axis_labels = ["Relevance", "Clarity", "Contribution", "Soundness", "Responsibility"]
        sns.heatmap(text_reviewer_scores_round_2_decimal, annot=True, cmap="YlGnBu", fmt='.2f', annot_kws={"size": 10}, linewidths=0.5, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        # Rotate the x-axis labels
        plt.xticks(rotation=30)
        # Rotate the y-axis labels to be horizontal
        plt.yticks(rotation=0)
        plt.xlabel("Meta-review criterion")
        plt.ylabel("Review criterion")
        
        
        # Save the plot
        plt.subplots_adjust(left=0.2, right=1.0, top=1.0, bottom=0.25)
        plt.savefig('reviewer_scores.png', dpi=300)
        plt.tight_layout()
        plt.close()

        # Save the plot in the html file
        binary_fc = open('reviewer_scores.png', 'rb').read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
        ext = 'png'
        dataurl = f'data:image/{ext};base64,{base64_utf8_str}'

        html_file.write("<img src="+dataurl+" alt='Reviewer scores' width='800'/>\n")
        os.remove('reviewer_scores.png')

    def plot_difference_of_scores_to_html(self, html_file):
        # Big font
        plt.rc('font', size=12)

        # For each criterion, plot the difference between the good and bad scores
        fig = plt.figure(figsize=(15,8))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        axes.append(fig.add_subplot(gs[0, 2]))
        axes.append(fig.add_subplot(gs[1, 0]))
        axes.append(fig.add_subplot(gs[1, 1]))
        for crit_i, criterion in enumerate(self.super_categories_for_text_reviewer):
            ax = axes[crit_i]

            pairs_of_scores = self.pairs_of_good_and_bad_scores[criterion]
            good_scores = [pair[0] for pair in pairs_of_scores]
            bad_scores = [pair[1] for pair in pairs_of_scores]
            difference_of_scores = np.array(good_scores) - np.array(bad_scores)

            # Perform a t-test to see if the difference is significant
            t_statistic, p_value = stats.ttest_rel(good_scores, bad_scores)
            # Create a boxplot of the difference
            bp = ax.boxplot(difference_of_scores, showfliers=False)
            # Add some random "jitter" to the x-axis
            x = np.random.normal(1, 0.04, size=len(difference_of_scores))
            ax.plot(x, difference_of_scores, 'r.', alpha=0.5)
            ax.set_ylim([-1, 1])

            # Add label and title
            ax.set_ylabel("Difference of good / bad")
            # Make the first letter of each word in the title uppercase
            ax.set_title(criterion.title())
            ax.set_xticks([])

            # Add t-statistic and p-value as text annotation
            text = f"t-statistic: {t_statistic:.2f}\np-value: {p_value:.2e}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

        # Save the plot
        # plt.subplots_adjust(left=0.1, right=1.0, top=1.0, bottom=0.1)
        
        plt.tight_layout()
        plt.savefig('difference_of_scores.png', dpi=300)
        plt.close()

        # Save the plot in the html file
        binary_fc = open('difference_of_scores.png', 'rb').read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
        ext = 'png'
        dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
        html_file.write("<img src="+dataurl+" alt='Difference of scores' width='1000'/>\n")
        os.remove('difference_of_scores.png')




    def write_paper_details_to_html_file(self, paper_details, html_file, anchor_name=''):
        """Write the paper details to html file.
        Args:
            paper_details: The paper details.
            html_file: The html file to write the paper details to.
        """
        paper_prompt = self.reviewer_metadata[self.reviewer_metadata['id'] == paper_details['id']]['prompt'].values[0]
        paper_type_good_or_bad = self.reviewer_metadata[self.reviewer_metadata['id'] == paper_details['id']]['good_or_bad'].values[0].split("_")[0]
        paper_review = paper_details['review']
        paper_meta_review_score = paper_details['meta_review_score']
        paper_meta_review_reason = paper_details['meta_review_reason']
        paper_meta_review_score_numpy_array = np.zeros((len(self.super_categories_for_text_reviewer),5))
        for i, criterion in enumerate(self.super_categories_for_text_reviewer):
            paper_meta_review_score_numpy_array[i] = paper_meta_review_score[criterion]

        with open(os.path.join(self.solution_dir, "reviewer", "papers", str(paper_details['id']) + ".json"), 'r') as f:
            paper_json = json.load(f)
        paper_title = paper_json[0]['text']
        paper_abstract = paper_json[1]['text']

        html_file.write("<style>body {font-size: 16pt;}</style>\n")

        # Write the paper's prompt to html file, the word Prompt is in bold
        html_file.write(f"&bull;<b><font size=\"+2\"> Prompt:</font></b> {paper_prompt}<br>")
        # Write the paper's type (good or bad) to html file
        html_file.write(f"&bull;<b><font size=\"+2\"> Type:</font></b> {paper_type_good_or_bad} paper<br>")
        # Add the full text link
        html_file.write(f"&bull;<font size=\"+2\"><a href=\"#{anchor_name}_full_text\"> Full paper text</a></font><br>")
        html_file.write("&bull;<b><font size=\"+2\"> Title and abstract:</font></b><br>")
        html_file.write(f"<br>\n")
        # Write the paper's title to html file, the title is in bold and font size is bigger, no word Title
        html_file.write(f"<center><b>{paper_title}</b></center><br>")
        # Write the paper's abstract to html file, the word Abstract is in bold
        html_file.write(f"<p>{paper_abstract}</p><br><br>")

        html_file.write("&bull;<span style='color:blue;'><b><font size=\"+2\"> Evaluation by meta reviewer: </font></b></span><br><br>")
        
        html_file.write(f"<span style='color:blue;'>Average score: {np.mean(paper_meta_review_score_numpy_array):.2f}</span><br><br>")

        # Plot the reviewer scores as a 4x5 table to html file.
        self.plot_reviewer_scores_to_html(paper_meta_review_score_numpy_array, html_file)
        html_file.write("<br>")

        # Details about each meta-review criterion, in smaller font size
        # - A - Rating: Is the score consistent with the text feed-back?
        # - B - Precision (clarity): Is the text feed-back precise (does it point to a specific reason of praise of criticism)?
        # - C - Correctness (soundness): Is the praise or criticism correct and well substantiated?
        # - D - Recommendation (contribution): Does the text feed-back provide detailed and actionable recommendations for improvement?
        # - E - Respectfulness (responsibility): Is the language polite and non discriminatory?
        html_file.write("<span style='color:blue;'><font size=\"+1\"><b>Rating: </b></span>Is the score consistent with the text feed-back?</font><br>")
        html_file.write("<span style='color:blue;'><font size=\"+1\"><b>Precision: </b></span>Is the text feed-back precise (does it point to a specific reason of praise of criticism)?</font><br>")
        html_file.write("<span style='color:blue;'><font size=\"+1\"><b>Correctness: </b></span>Is the praise or criticism correct and well substantiated?</font><br>")
        html_file.write("<span style='color:blue;'><font size=\"+1\"><b>Recommendation: </b></span>Does the text feed-back provide detailed and actionable recommendations for improvement?</font><br>")
        html_file.write("<span style='color:blue;'><font size=\"+1\"><b>Respectfulness: </b></span>Is the language polite and non discriminatory?</font><br>")


        html_file.write("<br><br>")
        html_file.write("&bull;<span style='color:blue;'><b><font size=\"+2\"> Meta-reviewer detailed feedback:</font></b></span><br><br>")
        # Write the paper's review, meta-review score and meta-review reason to html file in this format:
        # (The next texts are in black)
        # [Review criterion (e.g. Clarity, Contribution, Soundness, Responsibility)]: 
        # Score:[Review score]
        # Comment: [Review comment]
        # (The next texts are in blue)
        # [Meta-review criterion (e.g. Rating, Precision, Correctness, Recommendation, Respectfulness)]:
        # Score: [Meta-review score]
        # Reason: [Meta-review reason]
        pretty_super_categories_for_text_reviewer = ["Relevance", "Clarity", "Contribution", "Soundness", "Responsibility"]
        pretty_meta_criteria = ["Rating", "Precision", "Correctness", "Recommendation", "Respectfulness"]
        for i, criterion in enumerate(self.super_categories_for_text_reviewer):
            pretty_criterion = pretty_super_categories_for_text_reviewer[i]
            html_file.write(f"<b>{pretty_criterion}: </b>")
            html_file.write(f"{paper_review[criterion]['score']:.2f}<br>")
            html_file.write(f"{paper_review[criterion]['comment']}<br><br>")
            html_file.write("<span style='color:blue;'>Meta-review:</span><br>")
            for j, meta_review_score in enumerate(paper_meta_review_score[criterion]):
                meta_review_reason = paper_meta_review_reason[criterion][j]
                # Write the meta-review score and reason to html file in blue color to indicate meta-review
                html_file.write("&emsp;<b><span style='color:blue;'><font size=\"-1\">" + f"{pretty_meta_criteria[j]}: </font></span></b>")
                html_file.write("<span style='color:blue;'><font size=\"-1\">" + f"{meta_review_score:.2f}</font></span><br>")
                html_file.write("&emsp;<span style='color:blue;'><font size=\"-1\">" + f"{meta_review_reason}</font></span><br>")
            html_file.write("<br><br>")

        html_file.write(f"<h3 id=\"{anchor_name}_full_text\"><b>&bull; Full paper text:</b></h3><br>")
        # Write the paper's full text to html file
        for index, data in enumerate(paper_json):
            if index == 0:
                html_file.write(f"<h3>{data['heading']}: {data['text']}</h3><br><br>")
            else:
                html_file.write(f"<div><b>{data['heading']}</b><br>{data['text']}</div><br><br>")






        

        




        

        
