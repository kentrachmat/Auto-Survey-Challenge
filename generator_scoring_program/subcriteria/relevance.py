import json
import sys
import scipy
import torch
import bibtexparser
from sentence_transformers import SentenceTransformer
import logging
logging.basicConfig(level=logging.ERROR)

from .utils import *
from .base import *

MAX_TOKEN = 16_000
TRESHOLD = 0.5

class Relevance(Base):
    def __init__(self):
        super().__init__()
        self.model_name = 'paraphrase-MiniLM-L6-v2'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # get current path of this file
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(f"{current_real_dir}/models"):
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = SentenceTransformer(
                f"{current_real_dir}/models/{self.model_name}", device=self.device)

    def get_prompt(self, types, content, gt):
        if types == "title":
            return [
                {"role": "system", "content": "You are a helpful assistant who will help me review papers."},
                {"role": "user", "content": "You will receive a scientific paper title and a related prompt, both enclosed within XML tags. Your task is to evaluate the extent to which the title appropriately addresses the given prompt. Please provide your assessment score between 0 and 10, where 0 indicates no relevance and 10 indicates perfect relevance. Output your assessment exclusively in this JSON format: {\"score\" : \"...\", \"comment\" : \"...\"}" + f"<title>{content}</title> \n\n<prompt>{gt}</prompt>"},
            ]
        elif types == "abstract":
            return [
                {"role": "system", "content": "You are a helpful assistant who will help me review papers."},
                {"role": "user", "content": "You will receive a scientific paper abstract and a related prompt, both enclosed within XML tags. Your task is to evaluate the extent to which the abstract appropriately addresses the given prompt. Please provide your assessment score between 0 and 10, where 0 indicates no relevance and 10 indicates perfect relevance. Output your assessment exclusively in this JSON format: {\"score\" : \"...\", \"comment\" : \"...\"}" + f"<abstract>{content}</abstract> \n\n<prompt>{gt}</prompt>"},
            ] 

    def get_evaluation_section(self, participant, gt, types, model="gpt-3.5-turbo-16k"):
        success = False
        num_trials = 0
        while not success and num_trials < 5:
            try:
                answer = ask_chat_gpt(self.get_prompt(
                    types, participant, gt), model)
                answer = answer["choices"][0]["message"]["content"]
                return json.loads(answer)

            except Exception as e:
                print("Error: ", e)
                print("Retrying...")
                num_trials += 1
                success = False

    def get_evaluation_title(self, paper, generated_prompt):
        return self.get_evaluation_section(paper[0]['text'], generated_prompt, "title")
    
    def get_evaluation_abstract(self, paper, generated_prompt):
        return self.get_evaluation_section(paper[1]['text'], generated_prompt, "abstract")
 
    def get_comments_from_score(self, paper, score):
        if not self.reasons:
            return "TEST"
        conversation = [
            {"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
        conversation.append({"role": "user", "content":
                             "You are a helpful assistant who will help me review papers. " +
                             "Answer the following questions about paper below:\n" +
                             "For the question \"Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints?\", you have given an assessment score of " + f"{score:.2f}/1.0 . Elaborate on your assessment.\n" +
                             "\nPaper:\n" + paper})
        comment = ask_chat_gpt(conversation)[
            "choices"][0]["message"]["content"]
        return comment

    def get_evaluation(self, types, paper, generated_prompt):
        if types == "title":
            answer = self.get_evaluation_title(paper, generated_prompt)

        if types == "abstract":
            answer = self.get_evaluation_abstract(paper, generated_prompt)

        return {"score": answer['score'], "comment": answer['comment']}
    
    def get_combined_evaluation(self, _):
        pass