# Starting kit for PEER-REVIEWED JOURNAL OF AI-AGENTS CHALLENGE

---

# TODOs

* [x] Starting kit on Google Colab - [link](https://colab.research.google.com/drive/1dWgE9jE5TRFgTsaL6UWQoGHOS8lNEnQs?usp=sharing) || Need to be tested (clone part error)
* [x] Add another dummy way for code submission without chatGPT api key
* [x] Documentations on how to obtain ChatGPT api key for sample submission
* [ ] AI Author track: scoring program
* [ ] AI Reviewer track: scoring program
* [x] A script to reformat our data into competition data format

# Introduction

This competition is designed to propel advancements in the automation and fine-tuning of Large Language Models (LLMs), with an emphasis on automated prompt engineering. The primary application involves the generation of systematic review reports, overview papers, white papers, and essays that critically synthesize online information. The coverage spans multiple domains including Literary or philosophical essays (LETTERS), Scientific literature (SCIENCES), and topics surrounding the United Nations Sustainable Development Goals (SOCIAL SCIENCES).

# Task
Participants will work with challenge-provided keywords or brief prompts, transforming them into detailed prompts that elicit the production of concise articles or essays. These pieces, averaging four pages or about 2000 words, should include verifiable claims and precise references. Initially, the focus will be on text-only reports, with the anticipation of extending to multi-modality reports in future iterations of the challenge.

In this machine learning competition, participants will interface their models with an organizer-provided API. The model will function autonomously, using internet resources to tackle the tasks at hand. The contest simulates a peer-review process, with both the production and review of papers executed by AI systems. Participants will submit models capable of both generating and reviewing papers automatically. As a result, this competition will also foster the development of automated review systems.

The competition's assessment strategy relies on a peer-review principle, with papers generated by one model being evaluated by other models. Final decisions on paper acceptance, rejection, and awarding of best paper titles will be made by the organizers and jury, who will assess both the papers and their reviews.

# Data description

This challenge consists of 2 tracks:

- Generator track: The goal of this track is to generate systematic review papers according to the given prompts and instructions.
- Reviewer track: The goal of this track is to review research papers with the given criteria.

You are welcome to train your model using the any external data of your choice as long as you provide the neccesary API in your code prior to submission.

## 📂🗃️ Files

### Generator track

##### 📝 generator/prompts.csv

* `id`: A unique identifier for the paper
* `prompt`: A prompt with the instructions to generate the paper

##### 📝 generator/instructions.txt

The instructions of good practices to generate systematic review papers.

### Reviewer track

##### 📝 reviewer/metadata.csv

* `id`: A unique identifier for the paper
* `Responsibility`: A grade of how well the paper addresses potential risks or ethical issues and is respectful of human moral values, including fairness, and privacy, as a single number in the range \[0, 10\] included
* `Soundness`: A grade of how accurate the paper is and how well it presents accurate facts that are supported by citations of authoritative references, as a single number in the range \[0, 10\] included
* `Clarity`: A grade of how well the paper is written, with correct grammar and vocabulary, as well as how organized it is organized with meaningful sections and clearly explained concepts, as a single number in the range \[0, 10\] included
* `Contribution`: A grade of how much contribution the paper provides by introducing a comprehensive overview, comparing and contrasting a plurality of viewpoints, as a single number in the range \[0, 10\] included
* `Overall`: A overall grade of how good the paper is, as a single number in the range \[0, 10\] included
* `Confidence`: A score of how confidence the review is, as a single number in the range \[0, 10\] included

# Evaluation

### Metrics

#### Generator track

The evaluation metric for this track is the average of our ranking score.

For each generated paper and each criterion, we will compare them to our **well-generated** and **badly-generated** version. Participants will have access to this data in the Feedback Phase in the 📝 generator/papers folder which contains multiple folder associated with each unique `id` of the given prompts. 	Each folder consists of:

- `good_n.txt`: The Nth good paper that served as ground truth for the given prompt
- `bad_criterion_n.txt`: The Nth bad paper that served as bad examples of the specific criteria for the given prompt

The score of the generated paper will be either:

- `0`: if it is worse than both version in the given criterion
- `0.5`: if it is better than the **badly-generated** version and worse than the **well-generated** version.
- `1`: if it is better than both version in the given criterion

The process of comparing 2 papers differs in each phase:

- **Feedback phase & Development phase**: The comparison will be done by our baseline AI-reviewer
- **Final phase**: The comparison will be done by:
  - Experts in the field
  - Our baseline AI-reviewer
  - Peer-review by the AI-reviewer submitted in the Reviewer track

**FINAL DECISION:** The final ranking will be made by a jury of experts in each field.

#### Reviewer track

The evaluation metric for this track is the [Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) between the human reviews and the AI-reviewer's reviews for each criterion. The final score is the average of each criterion score.

The data used for evaluating is different in each phase:

- **Feedback phase & Development phase**: The dataset consists of human papers with actual human reviews only.
- **Final phase**: The dataset consists of AI-generated papers by all participants, and the human reviews will be done by our human experts in each field.

# Prerequisites

Install Anaconda and create an environment with Python 3.8 (RECOMMENDED)

### Usage:

- The file [README.ipynb](./README.ipynb) contains step-by-step instructions on how to create a sample submission  
- modify sample_code_submission to provide a better model or you can also write your own model in the jupyter notebook.
# References and credits

- Université Paris Saclay (https://www.universite-paris-saclay.fr/)
- ChaLearn (http://www.chalearn.org/)