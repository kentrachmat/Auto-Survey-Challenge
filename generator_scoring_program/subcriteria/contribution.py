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
THRESHOLD = 0.25

class Contribution(Base):
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
        if types == "abstract":

              # {"role": "user", "content": """You will receive a scientific paper abstract and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the given abstract appropriately summarizes the paper.  Please provide your assessment score between 0 and 10, where 0 is the lowest score. A good abstract should satisfy these checklists :
                # - The abstract should be a concise
                # - Background: What issues led to this work? What is the environment that makes this work interesting or important?
                # - Aim: What were the goals of this work? What gap is being filled?
                # - Approach: What went into trying to achieve the aims (e.g., experimental method, simulation approach, theoretical approach, combinations of these, etc.)? What was actually done?
                # - Results: What were the main results of the study (including numbers, if appropriate)?
                # Output your assessment exclusively in this JSON format: {\"score\" : \"...\", \"comment\" : \"...\"}""" + 
                # f"<abstract>{content}</abstract> \n\n<paper>{gt}</paper>"},
                
            return [
                {"role": "system", "content": "You are a helpful assistant who will help me review papers."},
                {"role": "user", "content": "You will receive a scientific paper abstract and title, both enclosed within XML tags. Your task is to evaluate the extent to which the abstract appropriately addresses the given title. Please provide your assessment score between 0 and 10, where 0 indicates no relevance and 10 indicates perfect relevance. Output your assessment exclusively in this JSON format: {\"score\" : \"...\", \"comment\" : \"...\"}" + f"<abstract>{content}</abstract> \n\n<title>{gt}</title>"},
            ]

        elif types == "conclusion":
            return [
                {"role": "system", "content": "You are a helpful assistant who will help me review papers."},
                {"role": "user", "content": """You will receive a scientific paper conclusion and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the conclusion appropriately highlight the main finding of the paper. Please provide your assessment score between 0 and 10, where 0 is the lowest score. A good conclusion should:

                - Restate the paper's topic and why it is important
                - Restate the paper's thesis/claim
                - Call for action or overview future research possibilities.

                Output your assessment exclusively in this JSON format: {"score" : "...", "comment" : "..."}
                """ + f"<conclusion>{content}</conclusion> \n\n<paper>{gt}</paper>"},
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

    def get_evaluation_coverage(self, paper, ref_coverage_paper):

        references = []
        si = 0
        sd = 0

        bib_database_ref = bibtexparser.loads(
            json.loads(ref_coverage_paper[0])[-1]['text'])
        for entry in bib_database_ref.entries:
            title = entry['title']
            references.append(title)

        references_participants = []
        bib_database = bibtexparser.loads(paper[-1]['text'])
        for entry in bib_database.entries:
            title = entry['title']
            # year = entry['year'] if 'year' in entry else ""
            references_participants.append(f"{title}")

        sentence_embeddings = self.model.encode(references_participants)

        for query in references:
            queries = [query]
            query_embeddings = self.model.encode(queries)

            distances = scipy.spatial.distance.cdist(
                query_embeddings, sentence_embeddings, "cosine")[0]

            results = [1-d for d in distances]
            results = sorted(results, reverse=True)
            if results[0] >= THRESHOLD:
                si += 1
            else:
                sd += 1

        coverage = si/(sys.float_info.epsilon + max(si, min(sd, 10)))
        if not self.reasons:
            comment = "TEST"
        else:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
            conversation.append({"role": "user", "content": f"You have given a score for the criteria of Coverage, now you need to provide the comments why did you gave the score. \n\n \
                                                                                                                                Score: {coverage:.2f}\n\nPaper:\n\n" + str(paper)})
            comment = ask_chat_gpt(conversation)[
                "choices"][0]["message"]["content"]

        return {"score": coverage, "comment": comment}

    def get_evaluation_abstract(self, paper):
        title = paper[0]['text']
        abstract = paper[1]['text']
        return self.get_evaluation_section(abstract, title, "abstract")

    def get_evaluation_conclusion(self, paper):
        conclusion_participant = paper[-2]
        conclusion = [] 

        for _, sect in enumerate(paper[1:]):
            prompt = self.get_prompt(
                "conclusion", conclusion_participant, "\n\n".join(conclusion))
            if num_tokens_from_messages(prompt) < MAX_TOKEN:
                conclusion.append(f"{sect['heading']}\n{sect['text']}")
                prompt = self.get_prompt(
                    "conclusion", conclusion_participant, "\n\n".join(conclusion))
                if num_tokens_from_messages(prompt) >= MAX_TOKEN:
                    del conclusion[-1]
                    break
            else:
                break

        if num_tokens_from_messages(prompt) >= MAX_TOKEN//4:
            model = "gpt-3.5-turbo-16k-0613"
        else:
            model = "gpt-3.5-turbo-0613"

        conclusion = "\n\n".join(conclusion)
        return self.get_evaluation_section(conclusion_participant, conclusion, "conclusion", model)

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

    def get_evaluation(self, types, paper, ref_coverage_paper):
        if types == "abstract":
            answer = self.get_evaluation_abstract(paper)

        if types == "conclusion":
            answer = self.get_evaluation_conclusion(paper)
 
        if types == "coverage":
            answer = self.get_evaluation_coverage(paper, ref_coverage_paper)

        return {"score": answer['score'], "comment": answer['comment']}
    
    def get_combined_evaluation(self, paper):
        pass