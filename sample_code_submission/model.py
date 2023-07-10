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
import tiktoken
import random


class model():
    def __init__(self):
        """
        This constructor is supposed to initialize data members. 
        """
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(
            current_real_dir, 'sample_submission_chatgpt_api_key.json')

        if isfile(target_dir):
            with open(target_dir, 'rb') as f:
                openai.api_key = json.load(f)['key']
        else:
            print("Warning: no api key file found.")

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
            success = False
            num_trials = 0
            while success == False and num_trials < 5:
                try:
                    body = []
                    conversation = self.conversation_generator("You are a helpful assistant who will help me generate survey papers around 2000 words.", f"You are a helpful assistant who will help me generate a survey paper with 2000 words. " + f"{prompts[i]}\nA good paper should:\n {instruction}")
                    body = json.loads(self.ask_chat_gpt(conversation, temperature=0.2*num_trials))
                    body_str = ""
                    for item in body:
                        body_str += item["heading"] + "\n" + item["text"] + "\n\n"

                    conversation = self.conversation_generator("You are a helpful assistant who will help me generate a paper abstract.", "You are a helpful assistant who will help me generate a paper abstract based on the text delimited with XML tags. Output the results exclusively in this JSON format: \{\"heading\":\"Abstract\",\"text\":\"....\"\}\n"f'<paper>{body_str}</paper>')
                    abstract = json.loads(self.ask_chat_gpt(conversation, temperature=0.2*num_trials))

                    conversation = self.conversation_generator("You are a helpful assistant who will help me generate a paper title.", "You are a helpful assistant who will help me generate a title based on the provided abstract delimited with XML tags." + f'<abstract>{abstract["text"]}</abstract>')
                    title = {"heading": "Title", "text": self.ask_chat_gpt(conversation, temperature=0.2*num_trials)}

                    conversation = self.conversation_generator("You are a helpful assistant who will help me generate paper references.", "You are a helpful assistant who will help me generate 10 references in BibTeX format based on the provided paper delimited with XML tags." + f'<paper>{body_str}</paper> \n' + "Output 10 references in a BibTeX format (without numbering and explanation) exclusively in this JSON format: \{\"heading\":\"References\",\"text\":\"....\"\}",)
                    refs = self.ask_chat_gpt(conversation, temperature=0.2*num_trials)
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
                    success = True
                except Exception as e:
                    print("Error:", e)
                    print("Retrying...")
                    num_trials += 1
            if num_trials == 5:
                print(f"Error: Exceeded maximum number of trials ({num_trials}). Returning an error paper.")
                generated_papers.append(json.dumps([{"heading": "Title", "text": "Error"}, {"heading": "Abstract", "text": "Error"}, {"heading": "Introduction", "text": "Error"}, {"heading": "Related Work", "text": "Error"}, {"heading": "Method", "text": "Error"}, {"heading": "Experiments", "text": "Error"}, {"heading": "Conclusion", "text": "Error"}, {"heading": "References", "text": "Error"}]))
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



            success = False
            num_trials = 0
            while success == False and num_trials < 5:
                try:
                    review_score = self.ask_chat_gpt(conversation, temperature=0.2*num_trials)
                    review_score = json.loads(review_score)
                    review_scores.append(review_score)
                    print("reviewing paper", i+1, "out of", len(papers))
                    success = True
                except Exception as e:
                    print("Error:", e)
                    print("Retrying...")
                    num_trials += 1
            
            if num_trials == 5:
                print(f"Error: Exceeded maximum number of trials ({num_trials}). Returning 0 scores.")
                review_scores.append({
                                "Responsibility": {
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                                "Soundness": {
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                                "Clarity":{
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                                "Contribution": {
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                                "Overall": {
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                                "Confidence": {
                                    "score": 0.0,
                                    "comment": "A comment."
                                },
                            })
        return review_scores

    def set_api_key(self, api_key):
        openai.api_key = api_key

    def conversation_generator(self, system, content):
        conversation = [{"role": "system", "content": system}]
        conversation.append({"role": "user", "content": content})
        return conversation

    def ask_chat_gpt(self, conversation, model="gpt-3.5-turbo-16k", temperature=0.0):
        if num_tokens_from_messages(conversation, model="gpt-3.5-turbo-0301") > 8_000:
            #num_tokens_from_messages() is greater than 8_000. Truncating conversation to 8_000 tokens
            while num_tokens_from_messages(conversation, model="gpt-3.5-turbo-0301") > 8_000:
                conversation[-1]["content"] = conversation[-1]["content"][:-1000]
        
        success = False
        number_trials = 0
        while not success:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    messages=conversation
                )
                success = True
            except Exception as e:
                number_trials += 1
                if number_trials > 5:
                    raise e

        return response.choices[0]['message']['content']

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
