import json
import sys
import scipy
import bibtexparser
from sentence_transformers import SentenceTransformer

from .utils import *
from .base import *

MAX_TOKEN = 16_000
TRESHOLD = 0.5


class Contribution(Base):
    def __init__(self):
        super().__init__()
        self.model_name = 'paraphrase-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)

    def get_prompt(self, types, content, gt):
        if types == "title":
            return [
                {"role": "system", "content": "You will receive a scientific paper title+abstract and a related prompt, both enclosed within XML tags. Your task is to evaluate the extent to which the title+abstract appropriately addresses the given prompt. Please provide your assessment score between 0 and 10, where 0 indicates no relevance and 10 indicates perfect relevance. Output your assessment exclusively in JSON format, like this: {\"score\" : \"...\", \"comment\" : \"...\"}"},
                {"role": "user", "content": f"<paper>{content}</paper> \n\n<prompt>{gt}</prompt>"},
            ]
        elif types == "abstract":
            return [
                {"role": "system", "content": """You will receive a scientific paper abstract and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the given abstract appropriately summarizes the paper.  Please provide your assessment score between 0 and 10, where 0 is the lowest score. A good abstract should satisfy these checklists :
                - The abstract should be a concise
                - Background: What issues led to this work? What is the environment that makes this work interesting or important?
                - Aim: What were the goals of this work? What gap is being filled?
                - Approach: What went into trying to achieve the aims (e.g., experimental method, simulation approach, theoretical approach, combinations of these, etc.)? What was actually done?
                - Results: What were the main results of the study (including numbers, if appropriate)?
                Output your assessment exclusively in JSON format, like this: {\"score\" : \"...\", \"comment\" : \"...\"}"""
                 },
                {"role": "user", "content": f"<abstract>{content}</abstract> \n\n<paper>{gt}</paper>"},
            ]

        elif types == "conclusion":
            return [
                {"role": "system", "content": """You will receive a scientific paper conclusion and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the conclusion appropriately highlight the main finding of the paper. Please provide your assessment score between 0 and 10, where 0 is the lowest score. Here's some guide of a good conclusion :

                - Restate your topic and why it is important
                - Restate your thesis/claim
                - Call for action or overview future research possibilities.

                Output your assessment exclusively in JSON format, like this: {"score" : "...", "comment" : "..."}
                        """},
                {"role": "user", "content": f"<conclusion>{content}</conclusion> \n\n<paper>{gt}</paper>"},
            ]

    def get_evaluation_section(self, participant, gt, types, model="gpt-3.5-turbo-16k"):
        success = False
        while not success:
            try:
                answer = ask_chat_gpt(self.get_prompt(
                    types, participant, gt), model)
                answer = answer["choices"][0]["message"]["content"]
                return json.loads(answer)
                success = True
            except Exception as e:
                print("Error: ", e)
                print("Retrying...")
                success = False

    def get_final_evaluation(self, c1, c2, c3):
        return (float(c1) + float(c2) + float(c3))/30

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
            if results[0] >= TRESHOLD:
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

    def get_evaluation_title(self, paper, generated_prompt):
        return self.get_evaluation_section(paper[0]['text'], generated_prompt, "title")

    def get_evaluation_abstract(self, paper):
        abstract = []
        for _, sect in enumerate(paper[2:]):
            prompt = self.get_prompt(
                "abstract", paper[1]['text'], "\n\n".join(abstract))
            if num_tokens_from_messages(prompt) < MAX_TOKEN:
                abstract.append(f"{sect['heading']}\n{sect['text']}")
                prompt = self.get_prompt(
                    "abstract", paper[1]['text'], "\n\n".join(abstract))
                if num_tokens_from_messages(prompt) >= MAX_TOKEN:
                    del abstract[-1]
                    break
            else:
                break

        if num_tokens_from_messages(prompt) >= MAX_TOKEN//4:
            model = "gpt-3.5-turbo-16k-0613"
        else:
            model = "gpt-3.5-turbo-0613"

        abstract = "\n\n".join(abstract)
        return self.get_evaluation_section(paper[1]['text'], abstract, "abstract", model)

    def get_evaluation_conclusion(self, paper):
        conclusion_participant = paper[-2]
        conclusion = []
        # for index, sect in enumerate(paper[2:]):
        #       if "concl" in sect['heading'].lower():
        #               conclusion_participant = paper[index]['text']
        #               break
        #       else:
        #               conclusion_participant = "no conclusion"

        for index, sect in enumerate(paper[1:]):
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
        conversation.append({"role": "user", "content": f"You have given a score for the criteria of Contribution, now you need to provide the comments why did you gave the score. \n\n \
                       Question: Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints? \n\nScore: {score:.2f}\n\nPaper:\n\n" + paper})
        comment = ask_chat_gpt(conversation)[
            "choices"][0]["message"]["content"]
        return comment

    def get_evaluation(self, types, paper, ref_coverage_paper, generated_prompt):
        if types == "title":
            answer = self.get_evaluation_title(paper, generated_prompt)

        if types == "abstract":
            answer = self.get_evaluation_abstract(paper)

        if types == "conclusion":
            answer = self.get_evaluation_conclusion(paper)

        if types == "final":
            json_title = self.get_evaluation_title(paper, generated_prompt)
            json_abstract = self.get_evaluation_abstract(paper)
            json_conclusion = self.get_evaluation_conclusion(paper)
            answer = self.get_final_evaluation(
                json_title['score'], json_abstract['score'], json_conclusion['score'])
            comment = self.get_comments_from_score(paper, answer)
            return {"score": answer['score'], "comment": comment}

        if types == "coverage":
            answer = self.get_evaluation_coverage(paper, ref_coverage_paper)

        return {"score": answer['score'], "comment": answer['comment']}