import openai
import json5 as json
import os
import random
from tqdm.auto import tqdm
from collections import defaultdict
from utils import *

from config import TRUNCATE

class BaselineReviewer:
    def __init__(self):
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_real_dir, 'scoring_program_chatgpt_api_key.json'), 'rb') as f:
            self.api_key = json.load(f)['key']

        openai.api_key = self.api_key

    def compare_papers(self, good_papers, bad_papers, prediction_paper, prediction_prompt, criteria):
        ''' Function to compare the solution and prediction papers
        Args:
            good_papers: list of good papers
            bad_papers: json file of bad papers
            prediction_paper: prediction paper
            prediction_prompt: prediction prompt
            criteria: criteria for comparison in json format 
                (for example: {'Clarity': 
                                    {'Correct grammar': '[0 if paper 1 is better, 1 if paper 2 is better]'}, 
                                    'Organization': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    'Explanation': '[0 if paper 1 is better, 1 if paper 2 is better]'},
                                    ...
                                'Contributions':
                                    {'Coverage': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                ...
                                }
                )
        Returns:
            comparison_result: comparison result in json format
                (for example: {'Clarity':
                                    {'Correct grammar': 0.0},
                                    'Organization': 0.0,
                                    'Explanation': 0.0},
                                    ...
                                'Contributions':
                                    {'Coverage': 0.0,
                                    ...
                                    }
                                ...
                                }
                )
            '''
    
        print("\nComparing good papers with prediction paper...")
        good_results = []
        for good_paper in tqdm(good_papers):
            good_result = self.compare_two_paper(good_paper, prediction_paper, json.dumps(criteria))
            good_results.append(good_result)
        # Combine good results
        combined_good_result = defaultdict(dict)
        for super_category, super_value in good_results[0].items():
            if isinstance(super_value, str):
                for good_result in good_results:
                    if super_category in combined_good_result:
                        combined_good_result[super_category] += int(good_result[super_category]) / len(good_results)
                    else:
                        combined_good_result[super_category] = int(good_result[super_category]) / len(good_results)  
            else:
                for sub_category, sub_value in super_value.items():
                    for good_result in good_results:
                        if super_category in combined_good_result and sub_category in combined_good_result[super_category]:
                            combined_good_result[super_category][sub_category] += int(good_result[super_category][sub_category]) / len(good_results)
                        else:
                            combined_good_result[super_category][sub_category] = int(good_result[super_category][sub_category]) / len(good_results)
        
        print("\nComparing bad papers with prediction paper...")
        combined_bad_result = defaultdict(dict)
        for super_category, super_value in tqdm(bad_papers.items()):
            if isinstance(super_value, list):
                for bad_paper in super_value:
                    result = self.compare_two_paper(bad_paper, prediction_paper, json.dumps({super_category: "[0 if paper 1 is better, 1 if paper 2 is better]"}))
                    if super_category in combined_bad_result:
                        combined_bad_result[super_category] += int(result[super_category]) / len(super_value)
                    else:
                        combined_bad_result[super_category] = int(result[super_category]) / len(super_value)
            else:
                for sub_category, sub_value in super_value.items():
                    for bad_paper in sub_value:
                        result = self.compare_two_paper(bad_paper, prediction_paper, json.dumps({super_category: {sub_category: "[0 if paper 1 is better, 1 if paper 2 is better]"}}))
                        if super_category in combined_bad_result and sub_category in combined_bad_result[super_category]:
                            combined_bad_result[super_category][sub_category] += int(result[super_category][sub_category]) / len(sub_value)
                        else:
                            combined_bad_result[super_category][sub_category] = int(result[super_category][sub_category]) / len(sub_value)

        # Combine good and bad results
        combined_result = defaultdict(dict)

        for super_category, super_value in combined_good_result.items():
            if isinstance(super_value, float):
                combined_result[super_category] = 0.5*super_value + 0.5*combined_bad_result[super_category]
            else:
                for sub_category, sub_value in super_value.items():
                    combined_result[super_category][sub_category] = 0.5*sub_value + 0.5*combined_bad_result[super_category][sub_category]

        print("\nCalculate soundness and confidence...")
        combined_result['soundness']['c1'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the references are legitimate)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        combined_result['soundness']['c2'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the references employed by the paper align well with the content it generates)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        combined_result['soundness']['c3'] = self.compare_one_paper(prediction_paper,'Evaluate the soundness of the paper (whether the paper does not randomly cite unrelated papers, maintaining a coherent academic narrative)','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        combined_result['confidence'] = self.compare_one_paper(prediction_paper,'Evaluate the confidence of the paper','[float number between [0,1] where 0.0 is the lowest and 1.0 is the highest]')
        
        print("Done calculating soundness and confidence")
        return combined_result
     
    def get_html_comments(self, generator_score, prediction_paper):
        combined_html = defaultdict(dict)

        print("Get html comments for each criterion...")
        for super_category, super_value in tqdm(generator_score.items()):
            if isinstance(super_value, float) or isinstance(super_value, int):
                    combined_html[super_category] = self.ask_chat_gpt_reason(prediction_paper, super_category, super_value)
            else:
                sub_criteria_gpt = self.ask_chat_gpt_reason(prediction_paper, str(super_value), None, "multiple")
                for sub_value in sub_criteria_gpt:
                    combined_html[super_category][sub_value['criterion']] = sub_value['reason']
        return dict(combined_html)
    
    def ask_chat_gpt_reason(self, prediction_paper, criterion, score, types="single"):
        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me evaluate a paper. You have given a score and now you need to provide a reason for the score."}]
        if types == "multiple":
            temp = '[{"criterion":..., "reason":...}, {"criterion":..., "reason":...}, ...]'

            conversation.append({"role": "user", "content":
            f"Please provide a short reason for each score where 0.0 is the lowest and 1.0 is the highest score for each criterion: {criterion}.\n\n" + \
            "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))] + \
            f"\n\nOutput a json file with a format {temp}"})

            result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
            return json.loads(result)
        else:
            conversation.append({"role": "user", "content":
            f"Please provide a short reason for the score {score:.2f} for the criterion: {criterion}:\n" + \
            "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))]})

            result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
            return str(result)
    
    def compare_one_paper(self, prediction_paper, description, criterion):
        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me evaluate a paper."}]
        conversation.append({"role": "user", "content":
        f"{description} the output of the template must be a single float value, no explanation:\nThe template:\n" + \
        criterion + \
        "Paper :\n" + prediction_paper[:(1000 if TRUNCATE else len(prediction_paper))]})

        result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
        return float(result)

    def compare_two_paper(self, paraphrased_paper, prediction_paper, criterion):
        """ Function to compare two papers
        Args:
            paraphrased_paper: paraphrased paper
            prediction_paper: prediction paper
            criterion: criterion for comparison in json format
                (for example: {'Clarity':
                                    {'Correct grammar': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                'Contributions':
                                    {'Coverage': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                ...
                                }
                )
        Returns:
            comparison_result: comparison result in json format
        """
        # 50% chance to swap the order of the papers
        if random.random() < 0.5:
            flipped = True
            paraphrased_paper, prediction_paper = prediction_paper, paraphrased_paper
        else:
            flipped = False

        paraphrased_paper_without_references = json.loads(paraphrased_paper)
        paraphrased_paper_without_references = [x for x in paraphrased_paper_without_references if x['heading'] != 'References']
        paraphrased_paper_without_references = json.dumps(paraphrased_paper_without_references)

        prediction_paper_without_references = json.loads(prediction_paper)
        prediction_paper_without_references = [x for x in prediction_paper_without_references if x['heading'] != 'References']
        prediction_paper_without_references = json.dumps(prediction_paper_without_references)

        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me compare papers."}]
        conversation.append({"role": "user", "content":
        f"Compare these 2 papers below and return the result using the template, no explanation:\nThe template:\n" + \
        criterion + \
        "The papers:\n{'Paper 1':\n" + paraphrased_paper_without_references[:(1000 if TRUNCATE else len(paraphrased_paper))] + ",\n'Paper 2':\n" + prediction_paper_without_references[:(1000 if TRUNCATE else len(prediction_paper))]+ '\n}'})

        # In case when the length of the prompt is too long, we will create a shorter prompt
        while num_tokens_from_messages(conversation) > 8_000:
            shorter_paraphrased_paper = json.loads(paraphrased_paper_without_references)
            shorter_paraphrased_paper = shorter_paraphrased_paper[:-1]
            shorter_paraphrased_paper = json.dumps(shorter_paraphrased_paper)

            shorter_prediction_paper = json.loads(prediction_paper_without_references)
            shorter_prediction_paper = shorter_prediction_paper[:-1]
            shorter_prediction_paper = json.dumps(shorter_prediction_paper)

            conversation = [{"role": "system", "content": "You are a helpful assistant who will help me compare papers."}]
            conversation.append({"role": "user", "content":
            f"Compare these 2 papers below and return the result using the template, no explanation:\nThe template:\n" + \
            criterion + \
            "The papers:\n{'Paper 1':\n" + shorter_paraphrased_paper + ",\n'Paper 2':\n" + shorter_prediction_paper+ '\n}'})

        result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
        # If the order of the papers is swapped, flip the result
        json_result = json.loads(result)
        if flipped:
            for super_category, super_value in json_result.items():
                if isinstance(super_value, str):
                    json_result[super_category] = str(1 - int(super_value))
                else:
                    for sub_category, sub_value in super_value.items():
                        json_result[super_category][sub_category] = str(1 - int(sub_value))
        return json_result 