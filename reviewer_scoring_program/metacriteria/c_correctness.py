
import tqdm
import json5 as json
import tqdm
import sys
import scipy

from .utils import *
from .base import *

class Correctness(Base):

	def get_prompt_for_evaluation(self, score, comment):
		prompt = f"Reviewer score: {score}, reviewer comment: {comment}\n" + \
				"Is the praise or criticism correct and well substantiated? Output your a single number in Likert scale from 1 to 3."
		return prompt

	def get_prompt_for_reason(self, score, comment, meta_review_score):
		prompt = f"Reviewer comment: {comment}" + \
				f"For the question: \"Is the praise or criticism correct and well substantiated?\", you gave an assessment score of {meta_review_score}/1.0 . Continue to criticize the review.\n"
		
		return prompt
	
	# def evaluate(self, score, comment):
	# 	prompt = [
	# 		{"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a grading and a reason for the score."},
	# 		{"role": "user", "content": \
	# 			"You are a meta-reviewer. One of your reviewers turned in feed-back for a paper. Is the praise or criticism correct and well substantiated?\n" + \
	# 			f"Score: {score} Comment: {comment}" + \
	# 			"Output your assessment in Likert scale from 1 to 3, exclusively in JSON format: {\"score\" : [1 for No, 2 for More-or-less, 3 for Yes]}\n" },
	# 	]
	# 	answer = ask_chat_gpt(prompt)["choices"][0]["message"]["content"]
	# 	meta_review_score = (float(json.loads(answer)["score"]) - 1) / 2
	# 	return meta_review_score

	# def get_reason(self, score, comment, meta_review_score):
	# 	prompt = [
	# 		{"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a reason for the score."},
	# 		{"role": "user", "content": \
	# 			"For the given score and comment:\n" + \
	# 			f"Score: {score} Comment: {comment}\n" + \
	# 			f"A meta-reviewer gave an assessment score of {meta_review_score} for the question: \"Is the praise or criticism correct and well substantiated?\". Please provide a short reason for the assessment."},
	# 	]
	# 	meta_review_comment = ask_chat_gpt(prompt)["choices"][0]["message"]["content"]

	# 	return meta_review_comment