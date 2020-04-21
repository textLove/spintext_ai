from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify

app = Flask(__name__)
cors = CORS(app)
from gensim.models import KeyedVectors
import urllib
from urllib.request import urlopen
from urllib.parse import urlencode
from bs4 import BeautifulSoup

from fake_useragent import UserAgent

import requests
import json
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import stopwords

# load the Stanford GloVe model
summary_ratio = 0.15
num_of_search_keywords = 5
top_n = 5
word2vec_output_file = "models/glove.6B.100d.txt.word2vec"
filename = word2vec_output_file
model = KeyedVectors.load_word2vec_format(filename, binary=False)
vocab_words =  list(model.vocab.keys())

def get_only_text(url):
  """
  return the title and text of the given URL
  """
  page = urlopen(url)
  soup = BeautifulSoup(page, "html.parser")
  text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
  return soup.title.text, text

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if all([w not in stopword_set, w in vocab_words])]))

    return cleaned_words

def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return round((1-cosine)*100,2)

def get_best_matches(summary, descriptions, links, n = 2):
  cosine_sims = []
  for desc in descriptions:
    cosine_sims.append(cosine_distance_wordembedding_method(summary, desc))
  if(n > len(cosine_sims)):
    n = len(cosine_sims)
  try:
    top_n_idx = np.argsort(cosine_sims)[-n:]
    top_n_scores = [cosine_sims[i] for i in top_n_idx]
    top_n_desc = [descriptions[i] for i in top_n_idx]
    top_n_links = [links[i] for i in top_n_idx]
    return top_n_idx,top_n_scores,top_n_desc,top_n_links
  except:
    print("exception while finding best_mathces")


#content_url = "https://www.kidsgen.com/stories/bedtime_stories/a_wise_parrot.htm"
#content_url = "https://www.kidsgen.com/stories/bedtime_stories/importance_of_guru.htm"
#content_url = "https://www.bedtimeshortstories.com/the-mouse-ghost"

def doProcess(title, text):
    doc_title = title
    toks = title.lower().split(" ")
    folder_name = "_".join(toks)
    import os

    folder_name = os.path.join("experiments", folder_name)

    try:
        # Create target Directory
        os.mkdir(folder_name)
        print("Directory " , folder_name ,  " Created ") 
    except FileExistsError:
        print("Directory " , folder_name ,  " already exists ")

    #save the hyper parameters
    text_file = open(folder_name + "/params.txt", "w")
    n = text_file.write(" summary_ratio=" + str(summary_ratio) + "\n num_of_search_keywords="+ str(num_of_search_keywords)  + "\n top_n=" + str(top_n))
    text_file.close()

    #save the original text
    text_file = open(folder_name + "/content.txt", "w")
    n = text_file.write(text)
    text_file.close()



    url = "https://api.smrzr.io/summarize?ratio=" + str(summary_ratio)

    payload = text.encode('utf-8')

    headers = {
    'Content-Type': 'text/plain'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    #print(response.text.encode('utf8'))
    summary = response.json()['summary']


    #save the summary text
    text_file = open(folder_name + "/summary.txt", "w")
    n = text_file.write(summary)
    text_file.close()

    url = "https://arcane-headland-47688.herokuapp.com:443/analyze"


    d = {"content": text}
    payload = json.dumps(d) #"{\n\t\"content\": \"Once upon a time there lived a parrot in a forest . He kept them in a cage and went to the place . They were served fruits and delicious food . They became the centre of attraction . When the king came to know about it , he ordered his men to leave the monkey in the forest . The bad days of the parrots were now over . The wise parrot clarified the situation to his younger brother saying that time never remains the same , and one should not be depressed by the temporary unfavorable changes.\"\n}"
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    ibmTxtAnalyzerResult = response.json()


    topKeywords = ibmTxtAnalyzerResult['keywords'][0:num_of_search_keywords]

    queryWords = []
    for keyword in topKeywords:
        queryWords.append(keyword['text'])
    print(" ".join(queryWords))
    ua = UserAgent()
    qstr1 = " ".join(queryWords)
    qstr1 = doc_title + " " + qstr1
    query = urllib.parse.quote_plus(qstr1) # Format into URL encoding
    number_result = 100
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    response = requests.get(google_url, {"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

    links = []
    titles = []
    descriptions = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
            
            # Check to make sure everything is present before appending
            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
        # Next loop if one element is not present
        except:
            continue


    vocab_words =  list(model.vocab.keys())

    q_txt = (" ".join(queryWords))
    q_txt = q_txt.encode('utf-8')
    #define search params:
    # _search_params = {
    #     'q': " ".join(queryWords),
    #     'num': 50,
    #     'safe': 'off',
    #     'fileType': '',
    #     'imgType': '',
    #     'imgSize': 'medium',
    #     'imgDominantColor': ''
    # }
    _search_params = {
        'key': "AIzaSyA6E5tDBr_-XwtP52CVLJVBq7U13IcmF4w",
        #'key': "AIzaSyAa8yy0GdcGPHdtD083HiGGx_S0vMPScDM",
        'cx': "002639010992359049478:yplzetgxswu",
        'q': q_txt,
        'num': 10,
        'safe': 'off',
        'imgSize': 'medium',
        'start': 1
    }

    url = "https://content.googleapis.com/customsearch/v1?" + urlencode(_search_params)
    print(url)

    payload = {}
    headers= {}

    response = requests.request("GET", url, headers=headers, data = payload)

    gimsearch_results = response.json()

    #save the google search results
    text_file_path = folder_name + "/gsearchResults.json"

    with open(text_file_path, 'w', encoding='utf-8') as f:
        json.dump(gimsearch_results, f, ensure_ascii=False, indent=4)

    imTitles = []
    imDescriptions = []
    imLinks = []
    file_extensions = ["jpg", "jpeg", "png", "gif"]
    for i in range(len(gimsearch_results['items'])):
        item = gimsearch_results['items'][i]
        try:
            title = item['title']
            desc = item['snippet']
            pagemap = item['pagemap']
            if title != '' and desc != '' and pagemap != '':
                im_path = pagemap['cse_image'][0]['src']
            
                if(im_path != ''):
                    ext = im_path.split(".")[-1]
                    if(ext in file_extensions):
                        imTitles.append(title)
                        imDescriptions.append(desc)
                        imLinks.append(im_path)
        except:
            print(i)
            continue

    (r_indexs,r_scores,r_descriptions,r_links) = get_best_matches(summary, imDescriptions, imLinks,n=top_n)
    print(summary)
    _obj1 = {
        "indexes": r_indexs.tolist(),
        "scores": r_scores,
        "descriptions": r_descriptions,
        "links": r_links
    }
    print(_obj1['scores'])
    #save the google search results
    text_file_path = folder_name + "/best_matches.json"

    with open(text_file_path, 'w', encoding='utf-8') as f:
        json.dump(_obj1, f, ensure_ascii=False, indent=4)

    return r_links

@app.route('/ping')
def hello():
    return "Pong!"

@app.route('/spintext', methods=['POST'])
def spintext():
    body = request.json
    print(body)
    resultLinks = []
    if("url" in body):
        url = body.get('url')
        title, text = get_only_text(url)
        resultLinks = doProcess(title, text)
    else:
        title = request.json.get('title')
        text = request.json.get('content')
        print(title)
        print(text)
        resultLinks = doProcess(title, text)
    return jsonify(resultLinks)

if __name__ == '__main__':
    app.run()