
import tqdm
import json5 as json
import tqdm
import sys
import scipy

from .utils import *
from .base import *

class Precision(Base):
	
	def get_prompt_for_evaluation(self, score, comment):
		prompt = f"Reviewer score: {score}, reviewer comment: {comment}\n" + \
				"Is the text feed-back precise (does it point to a specific reason of praise of criticism)? Output your a single number in Likert scale from 1 to 3."
		return prompt

	def get_prompt_for_reason(self, score, comment, meta_review_score):
		prompt = f"Reviewer comment: {comment}" + \
				f"For the question: \"Is the text feed-back precise (does it point to a specific reason of praise of criticism, be very demanding please and require specific examples)?\", you gave an assessment score of {meta_review_score}/1.0 . Continue to criticize the review.\n"
		
		return prompt
		
	# def get_prompt_for_reason(self, score, comment, meta_review_score):
	# 	prompt = [
	# 		{"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a reason for the score."},
	# 		{"role": "user", "content": \
	# 			"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a reason for the score. For the given score and comment:\n" + \
	# 			f"Score: {score} Comment: {comment}\n" + \
	# 			f"A meta-reviewer gave an assessment score of {meta_review_score} for the question: \"Is the text feed-back precise (does it point to a specific reason of praise of criticism, be very demanding please and require specific examples)?\". Please provide a short reason for the assessment."},
	# 	]
	# 	return prompt