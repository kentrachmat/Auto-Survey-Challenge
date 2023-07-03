
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

	def prompt_generator(self, types, content, gt):
		if types == "title":
			return [
			{"role": "system", "content":"You will receive a scientific paper title+abstract and a related prompt, both enclosed within XML tags. Your task is to evaluate the extent to which the title+abstract appropriately addresses the given prompt. Please provide your assessment score between 0 and 10, where 0 indicates no relevance and 10 indicates perfect relevance. Output your assessment exclusively in JSON format, like this: {\"score\" : \"...\", \"comment\" : \"...\"}"},
			{"role": "user", "content": f"<paper>{content}</paper> \n\n<prompt>{gt}</prompt>"},
			]
		elif types == "abstract":
			return [
			{"role": "system", "content":"""You will receive a scientific paper abstract and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the given abstract appropriately summarizes the paper.  Please provide your assessment score between 0 and 10, where 0 is the lowest score. A good abstract should satisfy these checklists :
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
			{"role": "system", "content":"""You will receive a scientific paper conclusion and the following sections, both enclosed within XML tags. Your task is to evaluate the extent to which the conclusion appropriately highlight the main finding of the paper. Please provide your assessment score between 0 and 10, where 0 is the lowest score. Here's some guide of a good conclusion :

		- Restate your topic and why it is important
		- Restate your thesis/claim
		- Call for action or overview future research possibilities.

		Output your assessment exclusively in JSON format, like this: {"score" : "...", "comment" : "..."}
			"""},
			{"role": "user", "content": f"<conclusion>{content}</conclusion> \n\n<paper>{gt}</paper>"},
			]
	
	def evaluate_section(self, participant, gt, types, model="gpt-3.5-turbo-16k"):
		answer = ask_chat_gpt(self.prompt_generator(types, participant, gt), model)["choices"][0]["message"]["content"]
		return json.loads(answer)

	def final_evaluation(self, c1,c2,c3):
		return (float(c1) + float(c2) + float(c3))/30
	
	def evaluate_coverage(self, paper, ref_coverage_paper):
		model_name = 'paraphrase-MiniLM-L6-v2'
		model = SentenceTransformer(model_name)

		references = []
		si = 0
		sd = 0

		for ref in ref_coverage_paper['references']:
			references.append(f" {ref['title']}")
		
		references_participants = []
		bib_database = bibtexparser.loads(paper[-1]['text'])
		for entry in bib_database.entries:
			title = entry['title']
			# year = entry['year'] if 'year' in entry else ""
			references_participants.append(f"{title}")

		sentence_embeddings = model.encode(references_participants)

		for query in references:
			queries = [query]
			query_embeddings = model.encode(queries)
			
			distances = scipy.spatial.distance.cdist(query_embeddings, sentence_embeddings, "cosine")[0]

			results = [1-d for d in distances]
			results = sorted(results, reverse=True)
			if results[0]>=TRESHOLD:
				si +=1
			else:
				sd +=1

		coverage = si/(sys.float_info.epsilon + max(si, min(sd,10)))

		conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers. You have given a score for the criteria of Coverage, now you need to provide the comments why did you gave the score."}]
		conversation.append({"role": "user", "content": f"Score: {coverage:.2f}\n\nPaper:\n\n" + paper})
		comment = self.askChatGpt(conversation)

		return {"score": coverage, "comment": comment}

	def evaluate_title(self, paper, generated_prompt):
		return self.evaluate_section(paper[0]['text'], generated_prompt, "title")
	
	def evaluate_abstract(self, paper):
		abstract = []
		for _, sect in enumerate(paper[2:]):
			prompt = self.prompt_generator("abstract", paper[1]['text'] , "\n\n".join(abstract))
			if num_tokens_from_messages(prompt) <  MAX_TOKEN:
				abstract.append(f"{sect['heading']}\n{sect['text']}")
				prompt = self.prompt_generator("abstract", paper[1]['text'] , "\n\n".join(abstract))
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
		return self.evaluate_section(paper[1]['text'], abstract, "abstract", model)
	
	def evaluate_conclusion(self, paper):
		conclusion_participant = ""
		conclusion = []
		for index, sect in enumerate(paper[2:]):
			if "concl" in sect['heading'].lower():
				conclusion_participant = paper[index]['text']
				break
			else:
				conclusion_participant = "no conclusion"

		for index, sect in enumerate(paper[1:]):
			prompt = self.prompt_generator("conclusion", conclusion_participant , "\n\n".join(conclusion))
			if num_tokens_from_messages(prompt) < MAX_TOKEN:
				conclusion.append(f"{sect['heading']}\n{sect['text']}")
				prompt = self.prompt_generator("conclusion", conclusion_participant , "\n\n".join(conclusion))
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
		return self.evaluate_section(conclusion_participant, conclusion, "conclusion", model)
	
	def comments(self, paper, score):
		conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers. You have given a score for the criteria of Contribution, now you need to provide the comments why did you gave the score."}]
		conversation.append({"role": "user", "content": f"Questiono: Does the answer provide a comprehensive overview, comparing and contrasting a plurality of viewpoints? \n\nScore: {score:.2f}\n\nPaper:\n\n" + paper})
		comment = self.askChatGpt(conversation)
		return comment
	
	def evaluate(self, paper, ref_coverage_paper, generated_prompt):
		paper = json.loads(paper)
		ref_coverage_paper = json.loads(ref_coverage_paper)
		
		# TITLE
		json_title = self.evaluate_title(paper, generated_prompt)

		# ABSTRACT
		json_abstract = self.evaluate_abstract(paper)

		# CONCLUSION
		json_conclusion = self.evaluate_conclusion(paper)

		# FINAL EVALUATION
		final = self.final_evaluation(json_title['score'] ,json_abstract['score'], json_conclusion['score'])

		# COVERAGE
		coverage = self.evaluate_coverage(paper, ref_coverage_paper)

		score = (final+coverage['score'])/2
		comment = self.comments(paper, score)

		return {"score": score, "comment": comment}
	
