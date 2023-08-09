# This is a configuration file for the sample solution.
# Feel free to change the default parameters below to your own needs.

# AI-Author task
# Number of papers to generate
NUM_PAPERS_TO_GENERATE = 5 # 1 for fast testing in < 5 minutes, 5 for a real submission



# AI-Reviewer task
# Number of set of papers to review (each set contains 20 generated papers from 1 prompt)
NUM_SETS_TO_REVIEW = 5 # 1 for fast testing in < 5 minutes, 5 for a real submission

# Evaluation mode:
# - 'fast' for fast testing in < 5 minutes. Your maximum score will be 0.5 / 1.0 because no LLM evaluation is performed.
# - 'full' for a real submission, with LLM evaluation
EVALUATION_MODE = 'full'
