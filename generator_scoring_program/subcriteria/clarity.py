from .base import *

from readability import Readability
import numpy as np

import argparse
import glob
import os
import pandas as pd
import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .utils import suppress_stdout_stderr

with suppress_stdout_stderr():
    from peft import (
        get_peft_config,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
        PeftType,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PeftConfig,
        PeftModel
    )

import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import transformers
transformers.logging.set_verbosity_error()
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from readability import Readability
import matplotlib.pyplot as plt
import textstat
import os
from scipy import stats

from .utils import ask_chat_gpt, custom_json_loads

import sys
sys.path.append("..")
from config import CLARITY_FULL_SCORE

import numpy as np
from readability import Readability


class Clarity(Base):
    def __init__(self):
        super().__init__()

        self.MAX_LENGTH = 1500  # SHOULD BE HUGE AND COMBINE WITH INT8 TRAINING IN THE FUTURE
        model_name_or_path = "microsoft/deberta-v3-large"
        # Get current path to this file
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(f"{current_real_dir}/models"):
            fbprize_peft_model_name_or_path = f"ktgiahieu/{model_name_or_path.split('/')[-1]}-peft-p-tuning-fbprize"
            fbprize_base_model_name_or_path = f"ktgiahieu/base-{model_name_or_path.split('/')[-1]}-peft-p-tuning-fbprize"
        else:
            fbprize_peft_model_name_or_path = f"{current_real_dir}/models/{model_name_or_path.split('/')[-1]}-peft-p-tuning-fbprize"
            fbprize_base_model_name_or_path = f"{current_real_dir}/models/base-{model_name_or_path.split('/')[-1]}-peft-p-tuning-fbprize"
        task = "mrpc"
        peft_type = PeftType.P_TUNING
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128, inference_mode=False)

        # ===== Tokenizers =====

        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        peft_config = PeftConfig.from_pretrained(
            fbprize_peft_model_name_or_path)
        self.inference_model = AutoModelForSequenceClassification.from_pretrained(
            fbprize_base_model_name_or_path, return_dict=True, num_labels=6, problem_type="regression")

        # Load the Lora model
        self.inference_model = PeftModel.from_pretrained(
            self.inference_model, fbprize_peft_model_name_or_path)
        self.inference_model.to(self.device)
        self.inference_model.eval()

    def tokenize_paper(self, paper):
        inputs = self.tokenizer.encode_plus(
            paper,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.MAX_LENGTH,
            padding=False,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor([v], dtype=torch.long)
        return inputs

    def get_kaggle_scores(self, paper):
        inputs = self.tokenize_paper(paper)
        # put the inputs to the device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
            # inference
        with torch.no_grad():
            outputs = self.inference_model(**inputs)
        outputs = outputs.logits.cpu().numpy().tolist()[0]
        return outputs

    def get_clarity_subscores(self, paper):
        """
        Clarity of the paper
        """

        # =============== Textstats ===============
        try:
            paper = self.json_or_text(paper)
            readability = Readability(paper)

            textstats_scores = [
                1 - (readability.flesch().score / 100),
                readability.dale_chall().score / 20,
                readability.ari().score / 20,
                readability.linsear_write().score / 20
            ]
        except:
            textstats_scores = [0] * 4

        if not CLARITY_FULL_SCORE:
            return textstats_scores[0], textstats_scores[1], textstats_scores[2]
        # =============== 6 Kaggle scores ===============
        kaggle_scores = self.get_kaggle_scores(paper)

        # =============== Combine ===============
        all_scores = np.array(kaggle_scores + textstats_scores).reshape(1, -1)

        # Logistic Regressor: Correct language
        clf = LogisticRegression()
        clf.coef_, clf.intercept_, clf.classes_ = (np.array([[1.12762367, 1.18921678, 0.87957839, 1.2248533, 1.47414413,
                                                              1.97531258, 0.1388447, 0.19495307, 0.35580047, 0.6719766]]),
                                                   np.array(
            [-31.39868405]),
            np.array([0., 1.]))
        correctlanguage_score = clf.predict_proba(all_scores)[0][1]

        # Logistic Regressor: Explanations
        clf = LogisticRegression()
        clf.coef_, clf.intercept_, clf.classes_ = (np.array([[1.50403552, 1.12114933, 0.72575994, 1.07584179, 1.2691045,
                                                              1.51088226, 0.20155372, 0.37622216, 0.18779782, 0.49807313]]),
                                                   np.array(
            [-28.76616597]),
            np.array([0., 1.]))
        explantions_score = clf.predict_proba(all_scores)[0][1]

        # Logistic Regressor: Organization
        clf = LogisticRegression()
        clf.coef_, clf.intercept_, clf.classes_ = (np.array([[0.3976214, 0.67363892, 0.92852696, 0.70761507, 0.6644566,
                                                              0.32860348, 0.09222614, 0.0184641, 0.11919781, 0.22703065]]),
                                                   np.array(
            [-15.03242901]),
            np.array([0., 1.]))
        organization_score = clf.predict_proba(all_scores)[0][1]

        return correctlanguage_score, explantions_score, organization_score

    def get_comments_from_subscores(self, paper,
                                    correctlanguage_score, explantions_score, organization_score):
        if not CLARITY_FULL_SCORE:
            return "Correct language comment", "Explanations comment", "Organization comment"
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant who will help me review papers."}]
        conversation.append({"role": "user", "content":
                             "You are a helpful assistant who will help me review papers. " + \
                             "Answer the following questions about the paper below in this JSON format:\{\n" +
                             "\"correct language\": \"For the question \"Is the paper written in good English, with correct grammar, and precise vocabulary?\", you have given an assessment score of " + f"{correctlanguage_score:.2f}/1.0 . Elaborate on your assessment.\",\n" +
                             "\"explanations\": \"For the question \"Are the concepts clearly explained, with short sentences?\", you have given an assessment score of " + f"{explantions_score:.2f}/1.0 . Elaborate on your assessment.\",\n" +
                             "\"organization\": \"For the question \"Is the paper well organized in meaningful sections and subsections?\", you have given an assessment score of " + f"{organization_score:.2f}/1.0 . Elaborate on your assessment.\"" +
                             "}" + "\nPaper:\n" + paper})
        success = False
        num_trials = 0
        while not success and num_trials < 5:
            try:
                comments = ask_chat_gpt(conversation)[
                    'choices'][0]['message']['content']
                comments = custom_json_loads(comments)
                correctlanguage_comment = comments['correct language']
                explantions_comment = comments['explanations']
                organization_comment = comments['organization']
                success = True
                return correctlanguage_comment, explantions_comment, organization_comment
            except Exception as e:
                print("An unexpected error occurred:", e)
                print("Retrying...")
                num_trials += 1
                success = False

    def get_evaluation(self, paper):
        paper = self.json_or_text(paper)
        correctlanguage_score, explantions_score, organization_score = self.get_clarity_subscores(paper)
        correctlanguage_comment, explantions_comment, organization_comment = self.get_comments_from_subscores(
            paper, correctlanguage_score, explantions_score, organization_score)

        correctlanguage = {"score": correctlanguage_score,
                           "comment": correctlanguage_comment}
        explantions = {"score": explantions_score,
                       "comment": explantions_comment}
        organization = {"score": organization_score,
                        "comment": organization_comment}
        return correctlanguage, explantions, organization

    def get_evaluation_correctlanguage(self, paper):
        correctlanguage, _, _ = self.get_evaluation(paper)
        return correctlanguage

    def get_evaluation_explantions(self, paper):
        _, explantions, _ = self.get_evaluation(paper)
        return explantions

    def get_evaluation_organization(self, paper):
        _, _, organization = self.get_evaluation(paper)
        return organization

    def get_combined_evaluation(self, paper):
        correctlanguage, explantions, organization = self.get_evaluation(paper)
        correctlanguage_score = correctlanguage['score']
        explantions_score = explantions['score']
        organization_score = organization['score']
        combined_score = (correctlanguage_score +
                          explantions_score + organization_score) / 3

        correctlanguage_comment = correctlanguage['comment']
        explantions_comment = explantions['comment']
        organization_comment = organization['comment']
        combined_comment = correctlanguage_comment + \
            explantions_comment + organization_comment

        return {"score": combined_score, "comment": combined_comment}