import openai
import json5 as json
import os
import random
from tqdm.auto import tqdm
from collections import defaultdict
from subcriteria import *

from config import *

class BaselineReviewer:
    def __init__(self):
        current_real_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_real_dir, 'scoring_program_chatgpt_api_key.json'), 'rb') as f:
            self.api_key = json.load(f)['key']

        openai.api_key = self.api_key

        self.responsibility = Responsibility()
        self.clarity = Clarity()
        self.contribution = Contribution()
        self.soundness = Soundness()
        self.relevance = Relevance()

    def compare_papers(self, good_papers, bad_papers, prediction_paper, prediction_prompt, criteria, human_paper):
        ''' Function to compare the solution and prediction papers
        Args:
            good_papers: list of good papers
            bad_papers: json file of bad papers
            prediction_paper: prediction paper
            prediction_prompt: prediction prompt
            criteria: criteria for comparison in json format 
                (for example: {'Clarity': 
                                    {'Correct grammar': '[0 if paper 1 is better, 1 if paper 2 is better]'}, 
                                    'Organization': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    'Explanation': '[0 if paper 1 is better, 1 if paper 2 is better]'},
                                    ...
                                'Contributions':
                                    {'Coverage': '[0 if paper 1 is better, 1 if paper 2 is better]',
                                    ...
                                    }
                                ...
                                }
                )
        Returns:
            comparison_result: comparison result in json format
                (for example: {'Clarity':
                                    {'Correct grammar': 0.0},
                                    'Organization': 0.0,
                                    'Explanation': 0.0},
                                    ...
                                'Contributions':
                                    {'Coverage': 0.0,
                                    ...
                                    }
                                ...
                                }
                )
            '''
        # If the prediction paper is {"Title: '', "Abstract": '', "Citations": '', "Introduction": '', "Related work": '', "Method": '', "Experiments": '', "Results": '', "Conclusion": ''}, or similar. Basically no values for the keys. 
        # Then we return the default values for the comparison result. which is 0 for all the criteria. and no comments.
        empty_string = True
        for dict in custom_json_loads(prediction_paper):
            if dict['text'] != '':
                empty_string = False
                break

        if empty_string:
            comparison_result = {}
            comparison_comment = {}
            for category, sub_criteria in criteria.items():
                if isinstance(sub_criteria, str):
                    comparison_result[category] = 0.0
                    comparison_comment[category] = "This is a test response. Please ignore."
                else:
                    comparison_result[category] = {}
                    comparison_comment[category] = {}
                    for sub_criterion, value in sub_criteria.items():
                        comparison_result[category][sub_criterion] = 0.0
                        comparison_comment[category][sub_criterion] = "This is a test response. Please ignore."

            # add soundness
            comparison_result['soundness'] = {'c1': 0.0, 'c2': 0.0, 'c3': 0.0}
            comparison_comment['soundness'] = {'c1': "This is a test response. Please ignore.", 'c2': "This is a test response. Please ignore.", 'c3': "This is a test response. Please ignore."}

            # add relevance
            comparison_result['relevance'] = {'title': 0.0, 'abstract': 0.0, 'citations': 0.0}
            comparison_comment['relevance'] = {'title': "This is a test response. Please ignore.", 'abstract': "This is a test response. Please ignore.", 'citations': "This is a test response. Please ignore."}
            return [comparison_result, comparison_comment]
            
        print("\nComparing good papers with prediction paper...")
        good_results = []
        good_results_comments = []

        for good_paper in tqdm(good_papers):
            good_result_answer = self.compare_good_papers(good_paper, prediction_paper, human_paper, prediction_prompt)
            good_result_answer
            good_result = good_result_answer[0]
            good_results_comment = good_result_answer[1]

            good_results.append(good_result)
            good_results_comments.append(good_results_comment)

        # DEBUG PURPOSES
        # good_results = [{'contribution': {'conclusion': 0, 'abstract': 0, 'title': 1, 'coverage': 0}, 'responsibility': 0, 'clarity': {'explanations': 0, 'correctlanguage': 0, 'organization': 0}, 'soundness': {'c1': 1, 'c2': 0, 'c3': 1}}, {'contribution': {'conclusion': 0, 'abstract': 1, 'title': 1, 'coverage': 0}, 'responsibility': 0, 'clarity': {'explanations': 0, 'correctlanguage': 0, 'organization': 0}, 'soundness': {'c1': 0, 'c2': 0, 'c3': 1}}, {'contribution': {'conclusion': 0, 'abstract': 0, 'title': 1, 'coverage': 0}, 'responsibility': 0, 'clarity': {'explanations': 1, 'correctlanguage': 1, 'organization': 1}, 'soundness': {'c1': 0, 'c2': 1, 'c3': 0}}]
        # good_results_comments = [{'contribution': {'conclusion': 'The conclusion appropriately highlights the main finding of the paper, which is the importance of understanding market failures, particularly credit constraints, in developing countries. It also mentions the need to explore market failures related to clean water in communities. However, it could have provided a more explicit call for action or overview of future research possibilities.', 'abstract': 'The abstract provides a concise summary of the paper, covering the background, aim, approach, and results of the study. It accurately highlights the contributions of Abhijit Banerjee, Esther Duflo, and Michael Kremer in the field of development economics and their experimental approach to alleviating global poverty. The abstract also mentions the specific areas of education, service delivery, and credit markets where their contributions have been significant. Overall, the abstract effectively summarizes the main points of the paper.', 'title': 'The title and abstract of the paper appropriately address the prompt by discussing the systematic survey of gradient approaches for guiding cell migration in neural tissue engineering. It specifically mentions the effectiveness of various scaffold cue presentation and methods to combine gradient approaches, with a focus on chemical, adhesive, mechanical, topographical, and electrical types of gradients.', 'coverage': 'The paper provides a comprehensive coverage of the contributions of Abhijit Banerjee, Esther Duflo, and Michael Kremer in the field of development economics. It discusses their experimental approach to alleviating global poverty and how it has transformed the field. The paper also highlights their pioneering contributions in understanding education, service delivery, and credit markets in developing countries. It provides examples of early experiments in these areas and discusses the challenges and mixed findings. The paper concludes by discussing the future directions of the field and the impact of their research on development policies. Overall, the paper covers the key aspects of their contributions and provides a thorough understanding of their work.'}, 'responsibility': 'The paper addresses potential risks and ethical issues by discussing the experimental approach used by Abhijit Banerjee, Esther Duflo, and Michael Kremer in development economics. It highlights the importance of understanding the educational production function, the challenges of service delivery in developing countries, and the impact of credit constraints and market failures. The paper also acknowledges the mixed findings of randomized trials and the need for further research in these areas. Overall, the paper demonstrates a respectful consideration of human moral values, including fairness and privacy, by discussing the implications of the experimental approach in addressing global poverty and informing development policies.', 'clarity': {'explanations': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high, and it effectively communicates the contributions of Banerjee, Duflo, and Kremer in the field of development economics.', 'correctlanguage': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high, and it effectively communicates the contributions of Banerjee, Duflo, and Kremer in the field of development economics.', 'organization': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high, and it effectively communicates the contributions of Banerjee, Duflo, and Kremer in the field of development economics.'}, 'soundness': {'c1': 'TEST', 'c2': 'TEST', 'c3': 'TEST'}}, {'contribution': {'conclusion': 'The conclusion appropriately highlights the main finding of the paper, which is the transformative impact of the experimental approach in development economics. It summarizes the contributions of Banerjee, Duflo, and Kremer in using randomized trials and field experiments to study a wide range of development issues. It also emphasizes the importance of understanding market failures and the microstructure of capital allocation in addressing the challenges of economic development. The conclusion effectively restates the main thesis of the paper and provides a call for future research and the application of research findings in development policy.', 'abstract': 'The abstract provides a concise summary of the paper. It clearly states the background and aim of the work, which is to provide an overview of the effectiveness of different gradient approaches in guiding cell migration in neural tissue engineering. The approach section describes the different types of gradients and provides examples of studies that have demonstrated their effectiveness. The results section summarizes the main findings of these studies. The abstract also mentions the potential of combining gradient approaches and the need for further research. Overall, the abstract effectively summarizes the key points of the paper.', 'title': 'The title and abstract of the paper appropriately address the prompt by discussing the systematic survey of gradient approaches for guiding cell migration in neural tissue engineering. It specifically mentions the effectiveness of various scaffold cue presentation and methods to combine gradient approaches, with a focus on chemical, adhesive, mechanical, topographical, and electrical types of gradients.', 'coverage': 'The paper provides comprehensive coverage of the topic. It starts with an introduction that explains the shift from macro to micro approaches in development economics and the role of randomized field experiments. It then discusses the background of macro development research in the 1980s and 1990s, highlighting the limitations of macroeconomic approaches. The paper goes on to explain the modern development economics and the experimental approach, emphasizing the importance of breaking down the development problem into component parts and using randomized field experiments. It provides early examples of the experimental approach in education, local governance and service delivery, and credit constraints and other market failures. The paper concludes by discussing the future directions of the field and the increasing use of experimental results in policy debates. Overall, the paper covers the key aspects of the topic in a comprehensive and informative manner.'}, 'responsibility': 'The paper addresses potential risks and ethical issues by discussing the experimental approach to development economics pioneered by Abhijit Banerjee, Esther Duflo, and Michael Kremer. It highlights how this approach has helped to break down the challenges of understanding economic development into component pieces and has led to an explosion of empirical research on various aspects of economic development. The paper also discusses the early examples of the experimental approach in the areas of education, governance of local service providers, and market failures. It acknowledges the mixed findings from some of the early studies and the challenges of sustaining improvements over time. The paper concludes by discussing the future directions of the experimental approach and its increasing impact on policy debates and development policies. Overall, the paper demonstrates a thorough understanding of the potential risks and ethical issues associated with the experimental approach in development economics and is respectful of human moral values, including fairness and privacy.', 'clarity': {'explanations': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high.', 'correctlanguage': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high.', 'organization': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and easy to understand. The paper is well organized into meaningful sections and subsections, which helps to guide the reader through the content. Overall, the clarity of the paper is high.'}, 'soundness': {'c1': 'TEST', 'c2': 'TEST', 'c3': 'TEST'}}, {'contribution': {'conclusion': 'The conclusion appropriately highlights the main finding of the paper, which is the transformative impact of the experimental approach in development economics. It also mentions the need for future research in exploring longer-term implications, market failures, and improving service provision. However, it could have provided a more explicit restatement of the thesis/claim and a stronger call for action or overview of future research possibilities.', 'abstract': 'The abstract provides a concise summary of the paper, covering the background, aim, approach, and results of the study. It accurately highlights the contributions of Abhijit Banerjee, Esther Duflo, and Michael Kremer in using randomized field experiments to understand and address development challenges. The abstract also mentions the specific areas of education, service delivery, and credit markets that were studied using the experimental approach. Overall, the abstract effectively summarizes the main points of the paper.', 'title': 'The title and abstract of the paper appropriately address the prompt by discussing the systematic survey of gradient approaches for guiding cell migration in neural tissue engineering. It specifically mentions the effectiveness of various scaffold cue presentation and methods to combine gradient approaches, with a focus on chemical, adhesive, mechanical, topographical, and electrical types of gradients.', 'coverage': 'The paper provides a comprehensive coverage of the contributions of Abhijit Banerjee, Esther Duflo, and Michael Kremer in the field of development economics. It discusses their experimental approach to alleviating global poverty and highlights how their approach has helped to advance the field of development economics. The paper also covers their pioneering contributions in understanding the challenges of education, service delivery, and credit markets in developing countries. It mentions the use of randomized field experiments and the Abdul Latif Jameel Poverty Action Lab, which has over 1,000 randomized evaluations completed or ongoing. The paper concludes by discussing the wide range of topics that have been studied using the experimental approach and the potential future directions for research in the field. Overall, the paper provides a comprehensive and informative coverage of the topic.'}, 'responsibility': 'The paper addresses potential risks and ethical issues by discussing the experimental approach to development economics. It highlights the importance of understanding the educational production function, the challenges of service delivery in developing countries, and the impact of credit constraints and market failures. The paper also acknowledges the mixed findings of randomized trials and the need for further research in these areas. Overall, the paper demonstrates a respectful consideration of human moral values, including fairness and privacy, by emphasizing the importance of evidence-based solutions to alleviate global poverty.', 'clarity': {'explanations': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and concise. The paper is well organized into meaningful sections and subsections, making it easy to follow the flow of information. Overall, the clarity of the paper is excellent.', 'correctlanguage': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and concise. The paper is well organized into meaningful sections and subsections, making it easy to follow the flow of information. Overall, the clarity of the paper is excellent.', 'organization': 'The paper is written in good English, with correct grammar and precise vocabulary. The concepts are clearly explained, and the sentences are short and concise. The paper is well organized into meaningful sections and subsections, making it easy to follow the flow of information. Overall, the clarity of the paper is excellent.'}, 'soundness': {'c1': 'TEST', 'c2': 'TEST', 'c3': 'TEST'}}]
        
        # Combine good results
        combined_good_result = defaultdict(dict)
        combined_good_result_comment = defaultdict(dict)
        for super_category, super_value in good_results[0].items():
            if isinstance(super_value, str) or isinstance(super_value, float) or isinstance(super_value, int):
                combined_reasons = []
                for index, good_result in enumerate(good_results):
                    combined_reasons.append(good_results_comments[index][super_category])
                    if super_category in combined_good_result:
                        combined_good_result[super_category] += int(good_result[super_category]) / len(good_results)
                    else:
                        combined_good_result[super_category] = int(good_result[super_category]) / len(good_results)  
                combined_good_result_comment[super_category] = self.ask_chat_gpt_combine_reason(" || ".join(combined_reasons))
            else:
                for sub_category, sub_value in super_value.items():
                    combined_reasons = []
                    for index, good_result in enumerate(good_results):
                        combined_reasons.append(good_results_comments[index][super_category][sub_category])
                        if super_category in combined_good_result and sub_category in combined_good_result[super_category]:
                            combined_good_result[super_category][sub_category] += int(good_result[super_category][sub_category]) / len(good_results)
                        else:
                            combined_good_result[super_category][sub_category] = int(good_result[super_category][sub_category]) / len(good_results)
                    combined_good_result_comment[super_category][sub_category] = self.ask_chat_gpt_combine_reason(" || ".join(combined_reasons))
        
        print("\nComparing bad papers with prediction paper...")

        combined_bad_result = defaultdict(dict)
        combined_bad_result_comment = defaultdict(dict)
        for super_category, super_value in tqdm(bad_papers.items()):
            if isinstance(super_value, list):
                comments = []
                for bad_paper in super_value:
                    answer = self.compare_bad_papers(super_category, bad_paper, prediction_paper, human_paper, prediction_prompt)
                    result = answer[0]
                    comments.append(answer[1])
                    if super_category == "soundness": # we could delete this

                        for c_i in ['c1', 'c2', 'c3']: 
                            if super_category in combined_bad_result and c_i in combined_bad_result[super_category]:
                                combined_bad_result[super_category][c_i] += int(result[c_i]) / len(super_value)
                            else:
                                combined_bad_result[super_category][c_i] = int(result[c_i]) / len(super_value)
                    else:
                        if super_category in combined_bad_result:
                            combined_bad_result[super_category] += int(result) / len(super_value)
                        else:
                            combined_bad_result[super_category] = int(result) / len(super_value)

                if super_category == "soundness": # we could delete this
                    for c_i in ['c1', 'c2', 'c3']:
                        combined_bad_result_comment[super_category][c_i] = self.ask_chat_gpt_combine_reason(" || ".join([value[c_i] for value in comments]))
                else:
                    combined_bad_result_comment[super_category] = self.ask_chat_gpt_combine_reason(" || ".join(comments))
            else:
                for sub_category, sub_value in super_value.items():
                    comments = []
                    for bad_paper in sub_value:
                        if super_category+f":{sub_category}" == "contribution:title":
                            continue
                        answer = self.compare_bad_papers(super_category+f":{sub_category}", bad_paper, prediction_paper, human_paper, prediction_prompt)
                        result = answer[0]
                        comments.append(answer[1])
                        if super_category in combined_bad_result and sub_category in combined_bad_result[super_category]:
                            combined_bad_result[super_category][sub_category] += int(result) / len(sub_value)
                        else:
                            combined_bad_result[super_category][sub_category] = int(result) / len(sub_value)
                    combined_bad_result_comment[super_category][sub_category] = self.ask_chat_gpt_combine_reason(" || ".join(comments))

        # Combine good and bad results
        combined_result = defaultdict(dict)
        combined_result_comment = defaultdict(dict)
        for super_category, super_value in combined_good_result.items():
            if isinstance(super_value, float) or isinstance(super_value, int):
                combined_result[super_category] = 0.5*super_value + 0.5*combined_bad_result[super_category]
                combined_result_comment[super_category] = self.ask_chat_gpt_combine_reason(combined_good_result_comment[super_category] + " || " + combined_bad_result_comment[super_category])
            else:
                for sub_category, sub_value in super_value.items():
                    combined_result[super_category][sub_category] = 0.5*sub_value + 0.5*combined_bad_result[super_category][sub_category]
                    combined_result_comment[super_category][sub_category] = self.ask_chat_gpt_combine_reason(combined_good_result_comment[super_category][sub_category] + " || " + combined_bad_result_comment[super_category][sub_category])

        # soundness
        prediction_paper = json.loads(prediction_paper)
        prediction_paper_soundness = self.soundness.get_evaluation(prediction_paper, prediction_prompt)
        evaluations = ['c1', 'c2', 'c3']
        for eval in evaluations:
            combined_result['soundness'][eval] = prediction_paper_soundness[eval]['score']
            combined_result_comment['soundness'][eval] = prediction_paper_soundness[eval]['comment']

        

        # change the order of the keys, relevance should be the first one
        final_combined_result = defaultdict(dict)
        final_combined_result_comment = defaultdict(dict)
        # relevance
        evaluations = ['title', 'abstract']
        for eval in evaluations:
            prediction_paper_relevance = self.relevance.get_evaluation(eval, prediction_paper, prediction_prompt)
            final_combined_result['relevance'][eval] = float(prediction_paper_relevance['score'])/10
            final_combined_result_comment['relevance'][eval] = prediction_paper_relevance['comment']

        eval_relevance = 'citations'
        final_combined_result['relevance'][eval_relevance] =prediction_paper_soundness[eval_relevance]['score']
        final_combined_result_comment['relevance'][eval_relevance] = prediction_paper_soundness[eval_relevance]['comment']

        for key in combined_result.keys():
            final_combined_result[key] = combined_result[key]
            final_combined_result_comment[key] = combined_result_comment[key]
        
        return [final_combined_result,final_combined_result_comment]
     
    def get_html_comments(self, generator_score, prediction_paper):
        combined_html = defaultdict(dict)

        print("Get html comments for each criterion...")
        for super_category, super_value in generator_score.items():
            if isinstance(super_value, float) or isinstance(super_value, int):
                    combined_html[super_category] = self.ask_chat_gpt_reason(prediction_paper, super_category, super_value)
            else:
                sub_criteria_gpt = self.ask_chat_gpt_reason(prediction_paper, str(super_value), None, "multiple")
                for sub_value in sub_criteria_gpt:
                    combined_html[super_category][sub_value['criterion']] = sub_value['reason']
        return dict(combined_html)
    
    def ask_chat_gpt_combine_reason(self, reasons):
        if not REASONS:
            return "This is a test response. Please ignore."

        conversation = [{"role": "system", "content": "You are a helpful assistant who will help me evaluate a paper."}]
        conversation.append({"role": "user", "content": f"Please combine these reasons into a single paragraph. \n\n REASONS: {reasons}"})
        result = ask_chat_gpt(conversation)["choices"][0]["message"]["content"]
        return result

    def compare_bad_papers(self, category, paraphrased_paper, prediction_paper, human_paper, prediction_prompt):
        paraphrased_paper = custom_json_loads(paraphrased_paper)
        prediction_paper = custom_json_loads(prediction_paper) 

        if category in ['clarity:explanations', 'clarity:correctlanguage', 'clarity:organization']:
            paraphrased_paper_clarity = self.clarity.get_combined_evaluation(paraphrased_paper)
            prediction_paper_clarity = self.clarity.get_combined_evaluation(prediction_paper)
            score = int(prediction_paper_clarity['score'] > paraphrased_paper_clarity['score'])
            comment = prediction_paper_clarity['comment'] if score else paraphrased_paper_clarity['comment']
            return [score, comment]
        
        elif "responsibility" in category:
            paraphrased_paper_responsibility = self.responsibility.get_evaluation(paraphrased_paper)
            prediction_paper_responsibility = self.responsibility.get_evaluation(prediction_paper)
            score_responsibility = int(prediction_paper_responsibility['score'] > paraphrased_paper_responsibility['score'])
            comment_responsibility = prediction_paper_responsibility['comment'] if score_responsibility else paraphrased_paper_responsibility['comment']
            return [score_responsibility, comment_responsibility]
        
        elif category in ['contribution:conclusion', 'contribution:abstract', 'contribution:coverage']:
            eval = category.split(":")[1]
            paraphrased_paper_contribution = self.contribution.get_evaluation(eval, paraphrased_paper, human_paper)
            prediction_paper_contribution = self.contribution.get_evaluation(eval, prediction_paper, human_paper)
            score = int(prediction_paper_contribution['score'] > paraphrased_paper_contribution['score'])
            comment = prediction_paper_contribution['comment'] if score else paraphrased_paper_contribution['comment']
            return [score, comment]
        
    def compare_good_papers(self, paraphrased_paper, prediction_paper, human_paper, prediction_prompt):
        paraphrased_paper = custom_json_loads(paraphrased_paper)
        prediction_paper = custom_json_loads(prediction_paper) 

        # # relevance
        # evaluations = ['title', 'abstract']
        # scores_relevance = {}
        # comments_relevance = {}
        # for eval in evaluations:
        #     paraphrased_paper_relevance = self.relevance.get_evaluation(eval, paraphrased_paper, prediction_prompt)
        #     prediction_paper_relevance = self.relevance.get_evaluation(eval, prediction_paper, prediction_prompt)
        #     score = int(prediction_paper_relevance[eval]['score']  > paraphrased_paper_relevance[eval]['score'] )
        #     comment = prediction_paper_relevance[eval]['comment'] if score else paraphrased_paper_relevance[eval]['comment']
            
        #     scores_relevance[eval] = score
        #     comments_relevance[eval] = comment

        # responsibility
        paraphrased_paper_responsibility = self.responsibility.get_evaluation(paraphrased_paper)
        prediction_paper_responsibility = self.responsibility.get_evaluation(prediction_paper)
        score_responsibility = int(prediction_paper_responsibility['score'] > paraphrased_paper_responsibility['score'])
        comment_responsibility = prediction_paper_responsibility['comment'] if score_responsibility else paraphrased_paper_responsibility['comment']

        # clarity
        evaluations = ['explanations', 'correctlanguage', 'organization']
        scores_clarity = {}
        comments_clarity = {}
        for eval in evaluations:
            paraphrased_paper_clarity = self.clarity.get_combined_evaluation(paraphrased_paper)
            prediction_paper_clarity = self.clarity.get_combined_evaluation(prediction_paper)
            score = int(prediction_paper_clarity['score'] > paraphrased_paper_clarity['score'])
            comment = prediction_paper_clarity['comment'] if score else paraphrased_paper_clarity['comment']
            
            scores_clarity[eval] = score
            comments_clarity[eval] = comment

        # contribution
        evaluations = ['conclusion', 'abstract', 'coverage']
        scores_contribution = {}
        comments_contribution = {}
        for eval in evaluations:
            paraphrased_paper_contribution = self.contribution.get_evaluation(eval, paraphrased_paper, human_paper)
            prediction_paper_contribution = self.contribution.get_evaluation(eval, prediction_paper, human_paper)
            score = int(prediction_paper_contribution['score'] > paraphrased_paper_contribution['score'])
            comment = prediction_paper_contribution['comment'] if score else paraphrased_paper_contribution['comment']
            
            scores_contribution[eval] = score
            comments_contribution[eval] = comment

        final_scores = {'contribution': scores_contribution, 
                'responsibility': score_responsibility, 
                'clarity': scores_clarity}
        
        final_comment = {'contribution': comments_contribution, 
                'responsibility': comment_responsibility, 
                'clarity': comments_clarity}

        return final_scores, final_comment