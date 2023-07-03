import os
import json
import openai
from os.path import isfile
from abc import ABC, abstractmethod

class Base(ABC):
	def __init__(self):
		current_real_dir = os.path.dirname(os.path.realpath(__file__))
		current_real_dir = os.path.dirname(current_real_dir)
		target_dir = os.path.join(current_real_dir, 'sample_submission_chatgpt_api_key.json')

		if isfile(target_dir):
			with open(target_dir, 'rb') as f:
				openai.api_key = json.load(f)['key']
		else:
			print("Warning: no api key file found.")
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
	def evaluate(self, paper):
		pass
