import time
import googleapiclient
from googleapiclient import discovery

from .base import *
from .utils import ask_chat_gpt

API_KEYS = "AIzaSyA7WSs8oKYvQtmaUUUkpyvesj2lFkLHi54"
ERROR_TIMEOUT = 60

#######################################################


class Responsibility(Base):
    def __init__(self):
        super().__init__()
        self.api_key = API_KEYS

    def get_score(self, paper):
        try:
            param = {"TOXICITY": {}}
        # "SEVERE_TOXICITY": {},"IDENTITY_ATTACK":{}, "INSULT": {}, "PROFANITY":{}, "THREAT":{}

            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=self.api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            analyse_request = {
                "comment": {"text": paper},
                "requestedAttributes": param,
                "spanAnnotations": True
            }

            response = client.comments().analyze(body=analyse_request).execute()

            for k in param.keys():
                param[k] = response["attributeScores"][k]["summaryScore"]["value"]
            return param

        except googleapiclient.errors.HttpError as err:
            if err.status_code == 429:
                time.sleep(ERROR_TIMEOUT)
                # print(
                #     f"sleeping for {ERROR_TIMEOUT} seconds, waiting for API quota to reset")
                return self.get_score(paper)

            elif err.status_code == 400:
                midpoint = len(paper) // 2
                part1 = self.get_score(paper[:midpoint])
                part2 = self.get_score(paper[midpoint:])
                averages = {}
                for criterion in part1:
                    total = part1[criterion] + part2[criterion]
                    average = total / 2
                    averages[criterion] = average
                return averages

    def get_comments_from_score(self, paper, score):
        if not self.reasons:
            return "TEST"
        conversation = [
            {"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
        conversation.append({"role": "user", "content":
                             "You are a helpful assistant who will help me review papers. " +
                             "Answer the following questions about paper below:\n" +
                             "For the question \"Does the paper address potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy?\", you have given an assessment score of " + f"{score:.2f}/1.0 . Elaborate on your assessment.\n" +
                             "\nPaper:\n" + paper})
        comment = ask_chat_gpt(conversation)[
            'choices'][0]['message']['content']
        return comment

    def get_evaluation(self, paper):
        """
        :param paper: a list of dictionaries, each with a heading and text or a string
        :return: a dictionary of the form {criterion: score} example : {'TOXICITY': 0.018723432}
        """

        paper = self.json_2_text(paper)
        score = 1 - float(self.get_score(paper)['TOXICITY'])
        comment = self.get_comments_from_score(paper, score)
        return {"score": score, "comment": comment}

    def get_combined_evaluation(self, paper):
        return self.get_evaluation(paper)
