
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
				"Is the text feed-back precise (does it point to a specific reason of praise of criticism)? You MUST ONLY output a single number in Likert scale from 1 to 3, no explanation is needed.\n"
		return prompt

	def get_prompt_for_reason(self, score, comment, meta_review_score):
		prompt = f"Reviewer comment: {comment}" + \
				f"For the question: \"Is the text feed-back precise (does it point to a specific reason of praise of criticism, be very demanding please and require specific examples)?\", you gave an assessment score of {meta_review_score}/1.0 . Continue to criticize the review.\n"
		
		return prompt
		