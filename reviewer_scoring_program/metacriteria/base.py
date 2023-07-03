from abc import ABC, abstractmethod

class Base(ABC):

	def jsonOrTxt(self, paper):
		if isinstance(paper, str):
			return paper
		elif isinstance(paper, list):
			return self.json2txt(paper)


	def json2txt(self, paper, offset=0):
		paper_str = ""
		for data in paper:
			paper_str += f"{data['heading']}\n{data['text']}\n\n"

		return paper_str[0:offset] if offset else paper_str
		
	# @abstractmethod
	# def evaluate(self, paper):
	# 	pass
