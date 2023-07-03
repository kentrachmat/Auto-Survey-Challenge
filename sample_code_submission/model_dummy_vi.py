"""
Sample predictive model.
You must supply at least 2 methods:
- generate_papers: calls the API to generate papers using the given prompts.
- review_papers: calls the API to review papers with the given instructions.
"""

import os
from os.path import isfile
import re 

from subcriteria import *

class model():
    def __init__(self):
        """
        This constructor is supposed to initialize data members. 
        """
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_real_dir, 'sample_submission_chatgpt_api_key.json')

        if isfile(target_dir):
            with open(target_dir, 'rb') as f:
                openai.api_key = json.load(f)['key']
        else:
            print("Warning: no api key file found.")

    def setApiKey(self, api_key):
        openai.api_key = api_key

    def conversation_generator(self, system, content):
        conversation = [{"role": "system", "content": system}]
        conversation.append({"role": "user", "content": content })
        return conversation

    def generate_papers(self, prompts, instruction):
        """
        Arguments:
            prompts: list of strings
            instructions: a string of instructions
        Returns:
            generated_papers: list of dictionaries
        """
        generated_papers = []
        for i in range(len(prompts)):
            try:
                body = []
                conversation = self.conversation_generator("You are a helpful assistant who will help me generate survey papers around 2000 words.", f"{prompts[i]} \n\ninstruction: {instruction}")
                body = json.loads(ask_chat_gpt(conversation)['choices'][0]['message']['content'])

                body_str = ""
                for item in body:
                    body_str += item["heading"] + "\n" + item["text"] + "\n\n"

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate an abstract based on the text delimited with XML tags. The output should be JSON formatted like \{\"heading\":\"Abstract\",\"text\":\"....\"\}", f'<paper>{body_str}</paper>')
                abstract = json.loads(ask_chat_gpt(conversation)['choices'][0]['message']['content'])

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate a title based on the provided abstract delimited with XML tags.", f'<abstract>{abstract["text"]}</abstract>')
                title = {"heading": "Title", "text": ask_chat_gpt(conversation)['choices'][0]['message']['content']}

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate 13 references in a BibTeX format (without numbering and explanation) based on the provided paper delimited with XML tags. The output should be JSON formatted like \{\"heading\":\"References\",\"text\":\"....\"\}", f'<paper>{body_str}</paper> \n\n Remember in a BibTeX format.')
                refs = ask_chat_gpt(conversation)['choices'][0]['message']['content']
                refs = re.sub(r"\n", r"\\n", refs)
                refs = re.sub(r"\\&", r"&", refs)
                refs = re.sub(r"\\~", r"~", refs)
                refs = re.sub(r"{\\\'\\i}", r"i", refs)
                refs = re.sub(r"{\\'e}", r"e", refs)
                refs = re.sub(r"{\\'o}", r"o", refs)
                refs = json.loads(refs)
                final = json.dumps([title]+[abstract]+body+[refs])
                print("Generated paper:", final)
                generated_papers.append(final)
                print("generated paper", i+1, "out of", len(prompts))
            except Exception as e:
                generated_papers.append(json.dumps({"heading": "Error", "text": "Error: the response is not a valid json string!"}))
                print("Error: the response is not a valid json string!", type(e), str(e))
                pass
        return generated_papers
	 
    def review_papers(self, papers, instruction, human, prompt):
        """
        Arguments:
            papers: list of strings
            instructions: a string of instructions
        Returns:
            review_scores: list of dictionaries of scores, depending on the instructions
        """
        responsibility = Responsibility()
        soundness = Soundness()
        clarity = Clarity()
        contribution = Contribution()

        review_scores = []
        for i in range(len(papers)):
            try:
                review_scores.append(
                    {
                        "Responsibility": responsibility.evaluate(papers[i]),
                        "Soundness": soundness.evaluate(papers[i]),
                        "Clarity": clarity.evaluate(papers[i]),
                        "Contribution": contribution.evaluate(papers[i], human[i], prompt[i])
                    }
                )
                print(review_scores)
                print("reviewing paper", i+1, "out of", len(papers))
            except:
                review_scores.append(
                    {
                        "Responsibility": {
                            "score": 0,
                            "comment": ""
                        },
                        "Soundness": {
                            "score": 0,
                            "comment": ""
                        },
                        "Clarity":{
                            "score": 0,
                            "comment": ""
                        },
                        "Contribution": {
                            "score": 0,
                            "comment": ""
                        }
                    }
                )
                print("Error: the response is not a valid json string.")
                pass

        return review_scores