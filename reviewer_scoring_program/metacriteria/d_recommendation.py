
import tqdm
import json5 as json
import tqdm
import sys
import scipy

from .utils import *
from .base import *

class Recommendation(Base):

	def get_prompt_for_evaluation(self, score, comment):
		prompt = f"Reviewer score: {score}, reviewer comment: {comment}\n" + \
				"Does the text feed-back provide detailed and actionable recommendations for improvement? You MUST ONLY output a single number in Likert scale from 1 to 3, no explanation is needed.\n"
		return prompt

	def get_prompt_for_reason(self, score, comment, meta_review_score):
		prompt = f"Reviewer comment: {comment}" + \
				f"For the question: \"Does the text feed-back provide recommendations for improvement that are both sufficiently detailed and actionable?\", you gave an assessment score of {meta_review_score}/1.0 . Continue to criticize the review.\n"
		
		return prompt