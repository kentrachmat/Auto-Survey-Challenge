import time
import googleapiclient
from googleapiclient import discovery

from .utils import *
from .base import *

API_KEYS = "AIzaSyA7WSs8oKYvQtmaUUUkpyvesj2lFkLHi54"
ERROR_TIMEOUT = 60

#######################################################

class Respectfulness(Base):
	def __init__(self):
		self.api_key = API_KEYS

	def evaluate_respectfulness_score(self, review):
		try:
			param = {"TOXICITY": {}}
	      	# "SEVERE_TOXICITY": {},"IDENTITY_ATTACK":{}, "INSULT": {}, "PROFANITY":{}, "THREAT":{}
			
			client = discovery.build(
				"commentanalyzer",
				"v1alpha1",
				developerKey=self.api_key,
				discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
				static_discovery=False,
			)

			analyse_request = {
				"comment": {"text": review},
				"requestedAttributes": param,
				"spanAnnotations" : True
			}
			
			response = client.comments().analyze(body=analyse_request).execute()

			for k in param.keys():
				param[k] = response["attributeScores"][k]["summaryScore"]["value"]
			return param
		
		except googleapiclient.errors.HttpError as err:
			if err.status_code == 429:
				time.sleep(ERROR_TIMEOUT)
				print(f"sleeping for {ERROR_TIMEOUT} seconds, waiting for API quota to reset")
				return self.evaluate_respectfulness_score(review)
			
			elif err.status_code == 400:
				midpoint = len(review) // 2
				part1 = self.evaluate_respectfulness_score(review[:midpoint])
				part2 = self.evaluate_respectfulness_score(review[midpoint:])
				
				averages = {}
				for criterion in part1:
					total = part1[criterion] + part2[criterion]
					average = total / 2
					averages[criterion] = average
				return averages

	def evaluate(self, score, comment):
		meta_review_score = 1 - self.evaluate_respectfulness_score(comment)['TOXICITY']
		return meta_review_score

	def get_prompt_for_reason(self, score, comment, meta_review_score):
		prompt = f"Reviewer comment: {comment}" + \
				f"For the question: \"Is the language polite and non discriminatory?\", you gave an assessment score of {meta_review_score}/1.0 . Continue to criticize the review.\n"

		return prompt

	def get_reason(self, score, comment, meta_review_score):
		prompt = [
			{"role": "system", "content":"You are a meta-reviewer who will help me evaluate a paper. You have given a score and a comment. Now you need to provide a reason for the score."},
			{"role": "user", "content": \
				"For the given score and comment:\n" + \
				f"Score: {score} Comment: {comment}\n" + \
				f"A meta-reviewer gave an assessment score of {meta_review_score} for the question: \"Is the language polite and non discriminatory?\". Please provide a short reason for the assessment."},
		]
		meta_review_comment = ask_chat_gpt(prompt)["choices"][0]["message"]["content"]

		return meta_review_comment