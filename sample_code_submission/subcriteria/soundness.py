from .base import *

import random
import time
import tqdm
import arxiv
import scipy
import requests
import numpy as np
import bibtexparser
from sentence_transformers import SentenceTransformer

headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
model_name = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

class Soundness(Base):

	def call_semantic(self, title):
		headers = {
			"x-api-key": "GYY0JXc5a1ax4IOsJ5giU5HTfRPplPoe8ZYddU4a"
		}
		url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=10&fields=url,authors,abstract,title,year"
		response = requests.get(url, headers=headers)
		if response.status_code == 200:
			return response.json()
		elif response.status_code == 429:
			time.sleep(5)
			return self.call_semantic(title)

	def call_scrapping(self, title):
		query = title.replace(" ","+")

		url = f"https://scholar.google.com/scholar?&q={query}+&hl=en&as_sdt=0,5&oq=o"
		doc = get_paperinfo(url)
		paper_tag,link_tag,abstract_tag,author_tag = get_tags(doc)

		papername = get_papertitle(paper_tag)
		year, publication , author = get_author_year_publi_info(author_tag)
		link = get_link(link_tag)
		abstract = get_abstract(abstract_tag)
		print(papername, abstract)

	def calculate_c1_c2(self, data_original):
		good = []
		bad = []
		bib_database = bibtexparser.loads(data_original[-1]['text'])

		for entry in bib_database.entries:
			if 'title' in entry and 'author' in entry and 'year' in entry:
				title = entry['title']
				authors = entry['author']
				year = entry['year']

				datas = self.call_semantic(title)
				if datas == None or datas['total'] == 0:
					print("[BAD 1: moving to scrapping] Title:", title)
					arr = self.call_scrapping(title)
					if arr != []:
						good.append({"title":arr[0],"abstract":arr[1]})
					else:
						print("[BAD 2: Scrapper not found] Title:", title)
						bad.append(title)
						continue

				try:
					title_list = []
					for data in datas['data']:
						title_list.append(data['title'])

					sentence_embeddings = model.encode(title_list)
					query_embedding = model.encode([title])
					distances = scipy.spatial.distance.cdist(query_embedding, sentence_embeddings, "cosine")[0]
					results = [1-d for d in distances]
					results = sorted(results, reverse=True)

					if results[0] >= 0.7:
						good.append(data)
					else:
						arr = self.call_scrapping(title)
					if arr != []:
						good.append({"title":arr[0],"abstract":arr[1]})
					else:
						bad.append(title)
					print("[BAD 3: SS not found] Title:", title)

				except Exception as e:
					bad.append(title)
					print("ERROR:", self.call_semantic(title, year))
					continue

			else:
				bad.append(title)
				print("[BAD 4: Format wrong] Title:", title)

		time.sleep(0.15)
		return (min(len(good), 10) - min(len(bad), 10))/10, good, bad

	def calculate_c2(self, data_original, good):
		ref_goods = []
		for g in good:
			ref_goods.append(f"{g['title']}. {g['abstract']}")

		sentence_embeddings = model.encode(ref_goods)

		queries = [f"{data_original[0]['text']}. {data_original[1]['text']}"]
		query_embedding = model.encode(queries)
		data_c2 = []

		distances = scipy.spatial.distance.cdist(query_embedding, sentence_embeddings, "cosine")[0]
		results = [1-d for d in distances]
		results = sorted(results, reverse=True)

		for res in results:
			data_c2.append(max(res,0))

		return np.mean(data_c2), sentence_embeddings

	def soundness_evaluation(self, c1,c2,c3):
		return (1+(c1*c2*c3))/2

	def evaluate(self): 
		return {"score":random.random(), "comment":"TEST"}