import os
import json
import openai
from os.path import isfile
from abc import ABC, abstractmethod
import sys
sys.path.append("..")   
from config import REASONS 

class Base(ABC): 
	def __init__(self):
		self.reasons = REASONS

	def flatten_dict(self, dictionary, parent_key='', sep='_'):
		flattened_dict = {}
		for key, value in dictionary.items():
			new_key = parent_key + sep + key if parent_key else key
			if isinstance(value, dict):
				flattened_dict.update(self.flatten_dict(value, new_key, sep))
			else:
				flattened_dict[new_key] = value
		return flattened_dict


	def json_or_text(self, paper):
		if isinstance(paper, str):
			return paper
		elif isinstance(paper, list):
			return self.json_2_text(paper)


	def json_2_text(self, paper, offset=0):
		paper_str = ""
		for data in paper:
			paper_str += f"{data['heading']}\n{data['text']}\n\n"

		return paper_str[0:offset] if offset else paper_str
		
	@abstractmethod
	def get_evaluation(self, paper):
		pass
