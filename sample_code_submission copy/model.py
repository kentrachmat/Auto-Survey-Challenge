"""
Sample predictive model.
You must supply at least 2 methods:
- generate_papers: generates a paper from a given prompt and instruction
- review_papers: generates a review score from a given paper and instruction
"""

import os
import re
import time
import json
from os.path import isfile
import openai
import random

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 15,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on any errors
            except errors as e:
                # Increment retries
                print(f"Error: {e}")
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # # Raise exceptions for any errors not specified
            except Exception as e:
                num_retries += 1
                pass

    return wrapper
 
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

    def set_api_key(self, api_key):
        openai.api_key = api_key

    def conversation_generator(self, system, content):
        conversation = [{"role": "system", "content": system}]
        conversation.append({"role": "user", "content": content })
        return conversation

    @retry_with_exponential_backoff
    def ask_chat_gpt(self, conversation, model="gpt-3.5-turbo-16k", temperature=0.0):
        response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=conversation
                )
        return response.choices[0]['message']['content']
    
    def generate_papers(self, prompts, instruction):
        """
        Arguments:
            prompts: list of prompts
            instructions: a string of instructions
        Returns:
            generated_papers: list of generated papers in a list
        """
        generated_papers = []
        for i in range(len(prompts)):
            try:
                body = []
                conversation = self.conversation_generator("You are a helpful assistant who will help me generate survey papers around 2000 words.", f"{prompts[i]} \n\ninstruction: {instruction}")
                body = json.loads(self.ask_chat_gpt(conversation))
                body_str = ""
                for item in body:
                    body_str += item["heading"] + "\n" + item["text"] + "\n\n"

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate an abstract based on the text delimited with XML tags. The output should be JSON formatted like \{\"heading\":\"Abstract\",\"text\":\"....\"\}", f'<paper>{body_str}</paper>')
                abstract = json.loads(self.ask_chat_gpt(conversation))

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate a title based on the provided abstract delimited with XML tags.", f'<abstract>{abstract["text"]}</abstract>')
                title = {"heading": "Title", "text": self.ask_chat_gpt(conversation)}

                conversation = self.conversation_generator("You are a helpful assistant who will help me generate 13 references in a BibTeX format (without numbering and explanation) based on the provided paper delimited with XML tags. The output should be JSON formatted like \{\"heading\":\"References\",\"text\":\"....\"\}", f'<paper>{body_str}</paper> \n\n Remember in a BibTeX format.')
                refs = self.ask_chat_gpt(conversation)
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
    
    def review_papers(self, papers, instruction):
        """
        Arguments:
            papers: list of papers
            instructions: a string of instructions
        Returns:
            review_scores: list of dictionaries of scores, depending on the instructions
        """

        review_scores = []
        for i in range(len(papers)):
            conversation = [{"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
            conversation.append({"role": "user", "content": instruction + json.dumps(papers[i])})

            review_score = self.ask_chat_gpt(conversation)
            
            try:
                review_score = json.loads(review_score)
                review_scores.append(review_score)
                print("reviewing paper", i+1, "out of", len(papers))
            except:
                review_scores.append({
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
                        },
                        "Confidence": {
                            "score": 0,
                            "comment": ""
                        }
                    }
                )
                print("Error: the response is not a valid json string.")
                pass
        return review_scores