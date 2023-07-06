from .base import *

import numpy as np
ALPHA = 0.5

class Confidence(Base):
	def get_calculate_score(self, dictionary): 
		flatten_dict = self.flatten_dict(dictionary)
		values = np.array(list(map(float, flatten_dict.values())))
		std = np.std(values)
		return np.exp(-ALPHA * std)
	
	def get_comments_from_score(self, paper, score):
		if not self.reasons:
			return "TEST"
		success = False
		while not success:
			try:
				conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
				conversation.append({"role": "user", "content": f"You have given a score for the criteria of Confidence, now you need to provide the comments why did you gave the score.\n\n \
																	Score: {score:.2f}\n\nPaper:\n\n" + paper})
				comment = self.askChatGpt(conversation)
				return comment
			except Exception as e:
				print("An unexpected error occurred:", e)
				print("Retrying...")
				success = False

	def get_evaluation(self, paper, dictionary):
		score = self.calculate(dictionary)
		comment = self.get_comments_from_score(paper, score)
		return {"score": score, "comment": comment}