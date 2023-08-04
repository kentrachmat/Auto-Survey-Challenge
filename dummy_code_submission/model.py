"""
Sample predictive model (dummy).
You must supply at least 2 methods:
- generate_papers: generate a dummy template text
- review_papers: generate a random score for each paper
"""

import json
 
class model():
    def __init__(self):
       pass

    def generate_papers(self, prompts, instruction):
        """
        Arguments:
            prompts: list of strings 
        Returns:
            generated_papers: list of dictionaries
        """
        generated_papers = []
        for i in range(len(prompts)):
            template = [{
                    "heading": "Title",
                    "text": ""
                },
                {
                    "heading": "Abstract",
                    "text": ""
                },
                {
                    "heading": "Introduction",
                    "text": ""
                },
                {
                    "heading": "Related work",
                    "text": ""
                },

                {
                    "heading": "Conclusion",
                    "text": ""
                },
                {
                    "heading": "References",
                    "text": ""
                }]
            print("Generated paper", i+1, "out of", len(prompts))
            generated_papers.append(json.dumps(template))
        return generated_papers
    

    def review_papers(self, papers, prompts, instruction):
        """
        Arguments:
            papers: list of strings 
            prompts: list of prompts used to generate the papers
            instruction: list of strings
        Returns:
            review_scores: list of dictionaries of scores, depending on the instructions
        """
        review_scores = []

        for i in range(len(papers)):
            review_score = {
                                "Relevance": {
                                    "score": 0.0,
                                    "comment": ""
                                },
                                "Responsibility": {
                                    "score": 0.0,
                                    "comment": ""
                                },
                                "Soundness": {
                                    "score": 0.0,
                                    "comment": ""
                                },
                                "Clarity":{
                                    "score": 0.0,
                                    "comment": ""
                                },
                                "Contribution": {
                                    "score": 0.0,
                                    "comment": ""
                                },
                                "Confidence": {
                                    "score": 0.0,
                                    "comment": ""
                                },
                            }
            print("reviewed paper", i+1, "out of", len(papers))
            review_scores.append(review_score)

        return review_scores