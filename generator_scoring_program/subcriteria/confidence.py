from .base import *

import numpy as np
ALPHA = 0.5

class Confidence(Base):
	def calculate(self, dictionary): 
		flatten_dict = self.flatten_dict(dictionary)
		values = np.array(list(map(float, flatten_dict.values())))
		std = np.std(values)
		return np.exp(-ALPHA * std)
	
	def comments(self, paper, score):
		conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers. You have given a score for the criteria of Confidence, now you need to provide the comments why did you gave the score."}]
		conversation.append({"role": "user", "content": f"Score: {score:.2f}\n\nPaper:\n\n" + paper})
		comment = self.askChatGpt(conversation)
		return comment

	def evaluate(self, paper, dictionary):
		score = self.calculate(dictionary)
		comment = self.comments(paper, score)
		return {"score": score, "comment": comment}