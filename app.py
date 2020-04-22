from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify
from flask import send_file

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
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize

import os
import math

# Server config
server_url = "https://localhost"

# load the Stanford GloVe model
summary_ratio = 0.15
num_of_search_keywords = 5
top_n = 5
word2vec_output_file = "models/glove.6B.100d.txt.word2vec"
filename = word2vec_output_file
model = KeyedVectors.load_word2vec_format(filename, binary=False)
vocab_words =  list(model.vocab.keys())

contractions = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}

known_entities = [
  "price",
  "battery life",
  "camera",
  "device compatibility",
  "overall performance",
  "market place",
  "charging",
  "finger print",
  "display clarity",
  "software platform",
  "design",
  "features",
  "product reliability",
  "front camera",
  "gaming",
  "apps",
  "lagging",
  "sound quality",
  "colour",
  "delivery",
  "call quality",
  "network",
  "ease of use",
  "memory",
  "picture quality",
  "touch sensitivity",
  "screen protection",
  "comfort",
  "build quality"
]

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

def load_df(fname):
    data_path = os.getcwd()+"/datasets/" + fname + ".csv";
    return pd.read_csv(data_path)

def create_folder(path):
    try:
        # Create target Directory
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        print("Directory " , path ,  " already exists ")
    return path;

def process_contradictions(text):
  for word in text.split():
    if word.lower() in contractions:
        text = text.replace(word, contractions[word.lower()])
  return text

def review_preprocess(review):
    result = process_contradictions(review)
    return result

def checkAnalsisExist(f_path):
    return os.path.exists(f_path + "/ibmTextAnalyser.json");

def loadIbmTxtAnalysis(f_path):
  print("loading from cache..")
  json_file_path = f_path + "/ibmTextAnalyser.json"
  with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())
  return contents

def loadSimilarPhraseCache():
  print("loading from cache..")
  json_file_path = os.getcwd() + "/similarPhrases.json"
  try:
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
  except:
      contents = {}
      saveSimilarPhraseCache({})
  return contents

def saveSimilarPhraseCache(obj):
  #save the google search results
    text_file_path = os.getcwd() + "/similarPhrases.json"
    with open(text_file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def loadGImageSearchResults(f_path):
  print("loading from cache..")
  json_file_path = f_path + "/gsearchResults.json"
  with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())
  return contents

def saveObject(f_path, obj):
  #save the google search results
    text_file_path = f_path + "/result.json"
    with open(text_file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def saveTxtAnalzeResult(f_path, result):
  #save the ibmTextAnalyzer output
  text_file_path = f_path + "/ibmTextAnalyser.json"
  with open(text_file_path, 'w', encoding='utf-8') as f:
      json.dump(result, f, ensure_ascii=False, indent=4)

def saveProcessedReviewTxt(f_path, review):
  #save the original text
    text_file = open(f_path + "/proc_review.txt", "w")
    n = text_file.write(review)
    text_file.close()

def saveReviewTxt(f_path, review):
  #save the original text
    text_file = open(f_path + "/review.txt", "w")
    n = text_file.write(review)
    text_file.close()

def watsonAnalyze(review):
    url = "https://arcane-headland-47688.herokuapp.com:443/analyze"
    d = {"content": review}
    payload = json.dumps(d) #"{\n\t\"content\": \"Once upon a time there lived a parrot in a forest . He kept them in a cage and went to the place . They were served fruits and delicious food . They became the centre of attraction . When the king came to know about it , he ordered his men to leave the monkey in the forest . The bad days of the parrots were now over . The wise parrot clarified the situation to his younger brother saying that time never remains the same , and one should not be depressed by the temporary unfavorable changes.\"\n}"
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    ibmTxtAnalyzerResult = response.json()
    return ibmTxtAnalyzerResult


def fuzzy_match(src, dest):
  score = fuzz.token_set_ratio(src, dest)
  return score

def isMatchWithKnownEntity(keyphrase):
  if(keyphrase in known_entities):
    return True
  scores = pd.Series(known_entities).apply(fuzzy_match,dest=keyphrase);
  max_score = scores.max()
  if(max_score > 60):
    return True
  return False

def extract_contextual_keywords(keywords, sentiment, reviewId):
  result = [];
  for keyword in keywords:
    obj = {}
    label = keyword['text']
    relevanceScore = keyword['relevance']
    obj['keyPhrase'] = label;
    obj['sentiment'] = sentiment;
    obj['relevanceScore'] = relevanceScore;
    obj['reviewIds'] = [reviewId];
    if(isSentimentLinked(label)):
      obj['ctx'] = True;
    else:
      obj['ctx'] = False;
    result.append(obj);
  return result

def isSentimentLinked(keyphrase):
    res = False
    text = word_tokenize(keyphrase)
    tags = dict(nltk.pos_tag(text))
    if('JJ' in tags.values() or 'JJS' in tags.values() or 'JJR' in tags.values()):
      res = True
    elif(isMatchWithKnownEntity(keyphrase)):
      res = True
    return res

def processReview(f_path, review, reviewIdx):
    saveReviewTxt(f_path, review)
    review = review_preprocess(review);
    saveProcessedReviewTxt(f_path, review)
    if(not checkAnalsisExist(f_path)):
      ibmTxtAnalyzerResult = watsonAnalyze(review)
    else:
      ibmTxtAnalyzerResult = loadIbmTxtAnalysis(f_path);
    if 'keywords' in ibmTxtAnalyzerResult.keys():
      saveTxtAnalzeResult(f_path, ibmTxtAnalyzerResult)
      senti_label = ibmTxtAnalyzerResult['sentiment']['document']['label']
      res = {
        "sentiment": ibmTxtAnalyzerResult['sentiment']['document']['label']
      }
      ctx_keywords_res = extract_contextual_keywords(ibmTxtAnalyzerResult['keywords'], senti_label, reviewIdx)
      saveObject(f_path,ctx_keywords_res)
      res['ctx_keywords'] = ctx_keywords_res;
      return res;
    else:
        return {}

def doProcessReviews(df,fname):
    folder_name = os.path.join("reviews", fname)
    create_folder(folder_name);
    result = []
    pos_reviews = 0;
    neg_reviews = 0;
    neutral_reviews = 0;
    other_reviews = 0
    for idx, review in df['Reviews'].items():
        print("-----idx---" + str(idx))
        f_path = folder_name + "/" + str(idx);
        create_folder(f_path) 
        res = processReview(f_path, review, idx);
        if("sentiment" in res.keys()):
            if(res['sentiment'] == 'positive'):
                pos_reviews = pos_reviews + 1
            elif(res['sentiment'] == 'negative'):
                neg_reviews = neg_reviews + 1
            elif(res['sentiment'] == 'neutral'):
                neutral_reviews = neutral_reviews + 1
            else:
                other_reviews = other_reviews + 1
            ctx_keywords_res = res['ctx_keywords']
            result = result + ctx_keywords_res
    f_obj = {
        "pos_reviews": pos_reviews,
        "neg_reviews": neg_reviews,
        "neutral_reviews": neutral_reviews,
        "other_reviews": other_reviews,
        "result": result
    }
    return f_obj


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

    if(not checkAnalsisExist(folder_name)):
      ibmTxtAnalyzerResult = watsonAnalyze(text)
      saveTxtAnalzeResult(folder_name, ibmTxtAnalyzerResult)
    else:
      ibmTxtAnalyzerResult = loadIbmTxtAnalysis(folder_name);


    topKeywords = ibmTxtAnalyzerResult['keywords'][0:num_of_search_keywords]

    queryWords = []
    for keyword in topKeywords:
        queryWords.append(keyword['text'])
    print(" ".join(queryWords))
    # ua = UserAgent()
    # qstr1 = " ".join(queryWords)
    # qstr1 = doc_title + " " + qstr1
    # query = urllib.parse.quote_plus(qstr1) # Format into URL encoding
    # number_result = 100
    # google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    # response = requests.get(google_url, {"User-Agent": ua.random})
    # soup = BeautifulSoup(response.text, "html.parser")

    # result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})

    # links = []
    # titles = []
    # descriptions = []
    # for r in result_div:
    #     # Checks if each element is present, else, raise exception
    #     try:
    #         link = r.find('a', href = True)
    #         title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
    #         description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
            
    #         # Check to make sure everything is present before appending
    #         if link != '' and title != '' and description != '': 
    #             links.append(link['href'])
    #             titles.append(title)
    #             descriptions.append(description)
    #     # Next loop if one element is not present
    #     except:
    #         continue


    # vocab_words =  list(model.vocab.keys())
    text_file_path = folder_name + "/gsearchResults.json"
    if(os.path.exists(text_file_path)):
        gimsearch_results = loadGImageSearchResults(folder_name)
    else:
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

def convertDFToJson(df1):
    d = df1.to_dict(orient='records')
    k = json.dumps(d)
    return k



def mergeReviewResults(reviews):
    # given more entries
    merged = []
    similarPhrases = loadSimilarPhraseCache()
    cur_similarPhrases = {}
    for review in reviews:
        cur_phrase = review['keyPhrase'];
        if(not cur_phrase in similarPhrases.keys()):
            similarPhrases[cur_phrase] = []
        if(not cur_phrase in cur_similarPhrases.keys()):
            cur_similarPhrases[cur_phrase] = []
        if(len(merged) == 0):
            merged.append(review)
        else:
            isMerged = False
            for m_review in merged:
                if(m_review['sentiment'] == review['sentiment']):
                    cur_text = review['keyPhrase'];
                    m_text = m_review['keyPhrase'];
                    if(m_text in similarPhrases[cur_phrase]):
                        if(not review['reviewIds'][0] in m_review['reviewIds']):
                            if(not m_text in cur_similarPhrases[cur_phrase]):
                                cur_similarPhrases[cur_phrase].append(m_text)
                            if(not m_text in cur_similarPhrases.keys()):
                                    cur_similarPhrases[m_text] = []
                                    cur_similarPhrases[m_text].append(cur_phrase)
                            else:
                                cur_similarPhrases[m_text].append(cur_phrase)
                            m_review['reviewIds'] = m_review['reviewIds'] + review['reviewIds']
                        isMerged = True
                    else:
                        dist = cosine_distance_wordembedding_method(cur_text, m_text);
                        if(dist > 60):
                            if(not review['reviewIds'][0] in m_review['reviewIds']):
                                m_review['reviewIds'] = m_review['reviewIds'] + review['reviewIds']
                            if(not m_text in similarPhrases[cur_phrase]):
                                similarPhrases[cur_phrase].append(m_text)
                            if(not m_text in cur_similarPhrases[cur_phrase]):
                                cur_similarPhrases[cur_phrase].append(m_text)
                                if(not m_text in cur_similarPhrases.keys()):
                                     cur_similarPhrases[m_text] = []
                                     cur_similarPhrases[m_text].append(cur_phrase)
                                else:
                                    cur_similarPhrases[m_text].append(cur_phrase)
                            isMerged = True
                            break;
            if(not isMerged):
                merged.append(review)
    saveSimilarPhraseCache(similarPhrases)
    # output merged entries
    f_res = {
        "similarPhrases": cur_similarPhrases,
        "merged": merged
    }
    return f_res

def postProcessReviewsResult(result):
    df = pd.DataFrame(result)
    postive_df_ctx = df[(df['sentiment'] == "positive") & (df['ctx'] == True)].sort_values('relevanceScore', ascending=False)
    negative_df_ctx = df[(df['sentiment'] == "negative") & (df['ctx'] == True)].sort_values('relevanceScore', ascending=False)
    postive_df_nctx = df[(df['sentiment'] == "positive") & (df['ctx'] == False)].sort_values('relevanceScore', ascending=False)
    negative_df_nctx = df[(df['sentiment'] == "negative") & (df['ctx'] == False)].sort_values('relevanceScore', ascending=False)
    total_ctx = len(postive_df_ctx) + len(negative_df_ctx);
    total_nctx = len(postive_df_nctx) + len(negative_df_nctx);
    pos_prop = (len(postive_df_ctx)/total_ctx);
    neg_prop = (len(negative_df_ctx)/total_ctx);
    print("Pos Len: " + str(len(postive_df_ctx)))
    print("Total Len: " + str(total_ctx))
    print("Pos prop: " + str(pos_prop))
    print("Neg prop: " + str(neg_prop))
    num_pos_ctx = round(pos_prop  * (10));
    num_neg_ctx = round(neg_prop  * (10));
    if(num_pos_ctx + num_neg_ctx >= 10):
        if(len(negative_df_ctx) != 0):
            return pd.concat([postive_df_ctx.head(num_pos_ctx), negative_df_ctx.head(num_neg_ctx)]).head(10)
        else:
            return pd.concat([postive_df_ctx.head(num_pos_ctx), negative_df_nctx.head(num_neg_ctx)]).head(10)
    else:
        reminging = 10 - num_pos_ctx + num_neg_ctx;
        num_pos_nctx = math.floor(pos_prop  * (reminging));
        num_neg_nctx = math.floor(neg_prop  * (reminging));
        return pd.concat([postive_df_ctx.head(num_pos_ctx), negative_df_ctx.head(num_neg_ctx) , postive_df_nctx.head(num_pos_nctx), negative_df_nctx.head(num_neg_nctx)]).head(10);

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

data_path = os.getcwd()+"/datasets/reviews"
list_of_files = {}

for filename in os.listdir(data_path):
    list_of_files[filename] = server_url + "/datasets/reviews/"+filename

@app.route('/list/datasets/reviews')
def getReviewDataset ():
    return jsonify(list_of_files)

@app.route('/datasets/reviews/<fname>')
def downloadFile (fname):
    path = data_path +  "/" + request.view_args['fname']
    return send_file(path, as_attachment=True)

@app.route('/reviewsummary', methods=['POST'])
def reviewSummary():
    body = request.json
    print(body)
    result = []
    if("url" in body):
        url = body.get('url')
        params = url.split("/");
        fname = params[len(params) - 1].replace(".csv", "")
        df = load_df('reviews/' + fname)
        new_columns = df.columns.values
        new_columns[0] = 'idx'
        df.columns = new_columns
        df = df.set_index('idx')
        total_reviews = len(df);
        print(len(df));
        f_obj = doProcessReviews(df,fname)
        result = f_obj['result']
        print(len(result))
        ob_result = mergeReviewResults(result);
        result = ob_result['merged']
        f_result = postProcessReviewsResult(result);
        j_result = convertDFToJson(f_result)
        t_result = {
            "totalReviews": total_reviews,
            "positvieReviwes": f_obj['pos_reviews'],
            "negativeReviews": f_obj['neg_reviews'],
            "neutralReviwes": f_obj['neutral_reviews'],
            "otherReviews": f_obj['other_reviews'],
            "similarPhrases": ob_result['similarPhrases'],
            "reviewSummary": j_result
        }
    return (t_result)

@app.route('/reviews', methods=['POST'])
def getReviews():
    body = request.json
    print(body)
    if("url" in body):
        url = body.get('url')
        params = url.split("/");
        fname = params[len(params) - 1].replace(".csv", "")
        df = load_df('reviews/' + fname)
        new_columns = df.columns.values
        new_columns[0] = 'idx'
        df.columns = new_columns
        df = df.set_index('idx')
        ids = body.get('ids')
        f_result = df[df.index.isin(ids)]['Reviews'].tolist()
    return jsonify(f_result)

@app.route('/reviews/similarphrases')
def getSimilarPhrases ():
    similarPhrases = loadSimilarPhraseCache()
    return jsonify(similarPhrases)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=443,ssl_context='adhoc')