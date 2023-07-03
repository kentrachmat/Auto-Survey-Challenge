#!pip install -q py-readability-metrics 
# !python -m nltk.downloader punkt

from .base import *

from readability import Readability
import numpy as np

from .utils import ask_chat_gpt
class Clarity(Base):
	def calculate(self, paper):
		"""
		Clarity of the paper
		"""
		try:
			readability = Readability(paper)

			textstats_scores = [
				readability.flesch().score / 30, 
				readability.dale_chall().score / 20,
				readability.ari().score / 20,
				readability.linsear_write().score / 20
			]
		except:
			textstats_scores = [0] * 4

		return np.mean(textstats_scores)
	
	def comments(self, paper, score):
		conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers. You have given a score for the criteria of Clarity (Is the paper written in good English, with correct grammar, and precise vocabulary? Is the paper well organized in meaningful sections and subsections? Are the concepts clearly explained, with short sentences), now you need to provide the comments why did you gave the score. Explain without the score"}]
		conversation.append({"role": "user", "content": f"Score: {score:.2f}\n\nPaper:\n\n" + paper})
		comment = ask_chat_gpt(conversation)['choices'][0]['message']['content']
		return comment
	
	def evaluate(self, paper):
		paper = self.json_or_text(paper)
		score = self.calculate(paper)
		comment = self.comments(paper, score)
		return {"score": score, "comment": comment}