from collections import OrderedDict
import requests
import bs4
import re
import tqdm
from collections import deque
from time import sleep
import random
import pandas as pd
import numpy as np

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import dalex as dx

porter = PorterStemmer()
lancaster = LancasterStemmer()
wordnet = WordNetLemmatizer()


class CustomStemmer():
    """Pipeline for stemming. You can deliver your custom stemmer, tokenizer, lemmatizer and stopwords and these will be used
    """
    def __init__(self, stemmer=LancasterStemmer, tokenizer=word_tokenize, lemmatizer=WordNetLemmatizer, custom_stopwords=stopwords.words('english')):
        """

        Args:
            stemmer (Class, optional): Any class with stem function, it will be instantiated. Defaults to LancasterStemmer.
            tokenizer (function, optional): Any function performing tokenization. Defaults to word_tokenize.
            lemmatizer (Class, optional): Any class with lemmatize function. Defaults to WordNetLemmatizer.
            custom_stopwords (List, optional): List with stopword to be removed. Defaults to nltk.corpus.stopwords.words('english').
        """
        self.stemmer = stemmer()
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer()
        self.custom_stopwords = custom_stopwords

    def process_text(self, text : str, stem=False, lemmatize=True):
        """Perform preprocessing of text stages are:
            tokenization -> removal of nonwords -> removing stopwords -> stemming or lemmatization

        Args:
            text (str): Text to be processed
            stem (bool, optional): to perform stemming. Defaults to False.
            lemmatize (bool, optional): to perform lemmatization. Defaults to True.

        Returns:
            str: processed text
        """
        words = self.tokenizer(text)
        final_words = []  # can't use set to preserve order
        for word in words:
            if re.match(r'^[A-Za-z]*$', word):
                final_words.append(word)
        cleaned = [
            word for word in final_words if word.lower() not in self.custom_stopwords]

        text = ' '.join(cleaned)
        if stem:
            text = self.stemmer.stem(text)
        elif lemmatize:
            text = self.lemmatizer.lemmatize(text)

        return text

    def process_corpus(self, data):
        """Process entire dataframe of text. This supports only DFs where text is stored in column called content.

        Args:
            data (Pandas.DataFrame): DataFrame it must posses content column with text to be processed.

        Returns:
            dict: Dictonary where keys are indices from DF and values are processed text from content column
        """
        processed = {index: self.process_text(
            element['content']) for index, element in data.iterrows()}
        return processed

    def generate_csv(self, processed, file_name="processed.csv"):
        """Saves output of process_corpus stage to csv.

        Args:
            processed (dict): dictonary returned from process_corpus function
            file_name (str, optional): Path where csv will be saved. Defaults to "processed.csv".
        """
        df = pd.DataFrame(processed, index=[0]).T
        df.columns = ['text']
        df.to_csv(file_name, columns=['text'])


class BFSScraper():
    """Class performing web scraping in BFS fashion
    """ 
    def __init__(self, n_to_visit : int):
        """

        Args:
            n_to_visit (int): How many links to visit on a page before moving forward 
            (This is to be set to omit sitations where page have 2000 links and we take links only from it)
        """
        self.n_to_visit = n_to_visit
        self.already_visited = set()
        self.q = deque()
        self.pages_with_error_response = {}
        self.pages = OrderedDict()  # We want to reproduce the order of visiting pages

    def get_unique_n_links(self, links):
        """This function returns at most n yet not visited links from given list of pages"""
        new_links = []
        candidates_ids = np.arange(len(links))
        np.random.shuffle(candidates_ids)  # To walk randomly
        for candidate_id in candidates_ids:  # Possibly all the links could be already visited, or we won't have n links
            if len(new_links) > self.n_to_visit:
                break
            link = "https://en.wikipedia.org" + links[candidate_id]['href']
            if link not in self.already_visited:
                new_links.append(link)

        return new_links

    def find_links(self, parsed_page):
        """Filter of useful wiki links

        Args:
            parsed_page (BeutifulSoup.soup): Page on which we want to find lists

        Returns:
            list: list of usefull wikipedia links
        """
        links = parsed_page.find_all(
            'a', attrs={'href': re.compile(r'^\/wiki\/(?!File)(?!Main_Page)\w*$')})  # To get only wikipedia articles, doen't take files nor something with :, ( etc. Don't go back to main page

        # To remove duplicates, probably not the most efficient way
        links = list(set(links))

        return links

    def process_one_link(self, link):
        """One step of traversal:
        Gets content of the page -> Find links -> find n unique links to be visited -> get content of currently visited page -> save all data into dictonary

        Args:
            link (str): link to be visited

        Returns:
           list:links to be visited next by BFS
        """
        response = requests.get(link)
        if response.status_code != 200:
            self.pages_with_error_response[link] = response.status_code
            return None

        parsed = bs4.BeautifulSoup(response.text, features="html.parser")
        found_links = self.find_links(parsed)
        n_not_visited_links = self.get_unique_n_links(found_links)
        content = "".join([p.getText()
                          for p in parsed.find(id="mw-content-text").select('p')])

        self.pages[link] = {}  # No OrderedDefaultDict :(
        self.pages[link]["content"] = content
        self.pages[link]["num_of_links"] = len(found_links)
        self.pages[link]["selected_links"] = n_not_visited_links

        self.already_visited.add(link)
        return n_not_visited_links

    def dummy_generator(self, n:int):
        """Just a fancy way to iterate until we have enough pages
        """
        while len(self.pages) < n:
            yield

    def generate_summary(self):
        """This creates text summary of BFS traversal in a format:
        visited_link - number of reasonable links: X
        Visited neighbours: 
                neighbour1
                neighbour2....
        """
        with open("summary.txt", 'w') as f:
            for link, page in self.pages.items():
                f.write(
                    f"{link} number of reasonable links: {page['num_of_links']}\n")
                f.write("Visited neighbours: \n")
                for neighbour in page["selected_links"]:
                    f.write(f"\t\t{neighbour}\n")
                f.write("\n\n")

    def generate_csv(self):
        df = pd.DataFrame(self.pages)
        df.to_csv('text.csv')

    def bfs(self, starting_link, n=1000):
        """Performs bfs traversal over a web until we visit n pages

        Args:
            starting_link (str):
            n (int, optional): how many pages to gather. Defaults to 1000.
        """
        self.q.append(starting_link)
        for _ in (pbar := tqdm.tqdm(self.dummy_generator(n))):
            link_to_scrap = self.q.popleft()
            links_to_visit = self.process_one_link(link_to_scrap)
            if links_to_visit is not None:  # Succesfull scraping of this particual pages and n neighbours gathered
                pbar.set_description(
                    f'{len(self.pages)} sites already collected')
                for link in links_to_visit:
                    self.q.append(link)

            sleep(random.random()*3)


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')

class User:
    """Class for simulating user of our retrieval system. Basically it just storest links visited and it's representation
    """
    def __init__(self,):
        self.history = [] # documents in tf-idf represetnation
        self.viewed_links = set() # visited links

    def add_to_history(self, link , tfidf_repr):
        if link not in self.viewed_links:
            self.history.append(tfidf_repr)
            self.viewed_links.add(link)

class Ranker:
    """Class that can perform ranking of documents given a user. It can do it based on the link (it simulates situation when user visits something and we recommend next things)
        or just based on history of the user.

        User can select between 3 representations: original TF-IDF, PCA or SVD
    """
    def __init__(self, data_path, pca_components=500):
        """
        Args:
            data_path (str): path to csv file with text, it must posses column named text
            pca_components (int, optional): number of components that will be created by PCA. Defaults to 500.
        """
        data = pd.read_csv(data_path, index_col=0)
        self.links = list(data.index)
        self.docs = list(data['text'])
        
        self.init_tfidf()
        self.init_svd()

        self.pca_components = pca_components
        self.init_pca()

        self.stemmer = CustomStemmer() # We need stemmer if one recommends from link

        self.scoring_methods = {
            "tfidf": self.tfidf_rank,
            "pca": self.pca_rank,
            "svd": self.svd_rank,
        }

    def init_tfidf(self):
        """Calculates tf-idf representation of documents and stores them as a data frame to reuse later
        """
        self.tfidf = TfidfVectorizer(use_idf=True, smooth_idf=False)
        trans = self.tfidf.fit_transform(self.docs) #! norm: l2 by default vectors are normalized
        self.trans_np = trans.toarray()
        self.dfTFIDF = pd.DataFrame(self.trans_np, index=np.arange(
            len(self.docs)), columns=self.tfidf.get_feature_names_out())

    def init_svd(self):
        """Performs CSV decomposition an then selects only concepts with singular values bigger than 1.
           This creates more compact representation of documents
        """
        u, s, vh = np.linalg.svd(self.trans_np.T, full_matrices=False) # calulating svd

        #np svd returns array of rank1 matrices
        u_c = np.concatenate(u[:, None])
        vh_c = np.concatenate(vh[:, None])

        #calculating diagonal matrix with singular values and it's inverse. Numpy returns flattened array
        s_shape = (s.shape[0], s.shape[0])
        s_c = np.zeros(s_shape)
        s_c[np.arange(s_shape[0]), np.arange(
            s_shape[0])] = s
        s_c_inv = np.zeros(s_shape)
        s_c_inv[np.arange(s_shape[0]), np.arange(
            s_shape[0])] = 1/s


        #Selecting only components having singular value bigger than 1
        how_many = (s_c > 1).sum()
        self.u_c_k = u_c[:, :how_many]
        self.s_c_inv_k = s_c_inv[:how_many, :how_many]
        vh_c_k = vh_c[:how_many, :] # This matrix represents all documents

        #Conversion to DataFrame so that apply operation is easy
        self.dfSVD = pd.DataFrame(
            vh_c_k.T, index=np.arange(len(self.docs)))

    def init_pca(self, printing=False):
        """Generates PCA representation of documents, Allows for more compact representation

        Args:
            printing (bool, optional): If set to True information about explained variance will be printed . Defaults to False.
        """
        self.pca = PCA(n_components=self.pca_components)
        self.pca.fit(self.trans_np)
        if printing:
            print(f"Using {self.pca_components} directions we were able to explain: {self.pca.explained_variance_ratio_.sum()}% of variance using PCA")
        
        #Transform all documents and create dataframe
        docs_transformed = self.pca.transform(self.trans_np)
        self.dfPCA = pd.DataFrame(
            docs_transformed, index=np.arange(len(self.docs)))

    def tfidf_query(self, query):
        """Transforms query to tf-idf representation"""
        return self.tfidf.transform([query]).toarray()[0] # Transform query

    def tfidf_rank(self, query_tfidf):
        """Performs rank using tfidf representation"""
        return 1-self.dfTFIDF.apply(lambda x: cosine(x, query_tfidf), axis=1) # Compute cosine similarity

    def pca_rank(self, query_tfidf):
        """Performs rank using PCA representation"""
        query_pca = self.pca.transform(query_tfidf[None, :]) # Transform query
        return 1-self.dfPCA.apply(lambda x: cosine(x, query_pca[0]), axis=1) # Compute cosine similarity

    def svd_rank(self, query_tfidf):
        """Performs rank using SVD representation"""
        query_svd = query_tfidf @ self.u_c_k @ self.s_c_inv_k
        return 1-self.dfSVD.apply(lambda x: cosine(x, query_svd), axis=1) # Compute cosine similarity


    def get_page(self, link):
        """Just gathers content of given wikipedia article

        Args:
            link (str): link to wiki article

        Raises:
            Exception: If response code is different than 2000

        Returns:
            str: text of the document
        """
        try:
            i = self.links.index(link)
            return self.docs[i]         
        except ValueError:
            response = requests.get(link)
            if response.status_code != 200:
                raise Exception("Failed to gather data")

            parsed = bs4.BeautifulSoup(response.text, features="html.parser")
        
            return "".join([p.getText()
                            for p in parsed.find(id="mw-content-text").select('p')]).strip()

    def show_recommendations(self, user: User, scores, top_n, scored_link=None):
        """Given calculated scores it just present ranking to the user. It takes care about amount of links shown and also it excludes links already visited by user

        Args:
            user (User): User for whom we perform recommendation
            scores (array):
            top_n (int): How many links to show
            scored_link (_type_, optional): Link on which scoring was performed to exclude it. Defaults to None.
        """
        scores_df = {
            "link" : self.links,
            "score": list(scores)
        }
        scores_df = pd.DataFrame(scores_df).sort_values(by="score", ascending=False)

        shown = 0
        i = 0
        while shown < top_n:
            link, score = scores_df.iloc[i]
            if link not in user.viewed_links and link != scored_link:
                print(f"{link}, score: {score}")
                shown +=1
            i+=1

    def recommend_based_on_history(self, user: User, model='tfidf', top_n = 10):
        """Calculate score between user history and our corpora of textts and then shown recommendations

        Args:
            user (User): 
            model (str, optional): Which model to chose can be pcs, svd or tfidf. Defaults to 'tfidf'.
            top_n (int, optional): . Defaults to 10.
        """
        if user.history is None:
            print("History of this user is empty. Start from rank_based_on_link")
            return None
        
        scores = [self.scoring_methods[model](doc) for doc in user.history]
        scores = np.round((np.array(scores).mean(axis=0)), decimals=4)
        self.show_recommendations(user, scores, top_n)

    
    def rank_based_on_link(self, user: User, link, model='tfidf', top_n=10):
        """Shows pages similar to given link. Additionally ads this link to visiting history

        Args:
            user (User): 
            link (str): link to wikipedia article
            model (str, optional): pca,svd or tfidf. Defaults to 'tfidf'.
            top_n (int, optional): . Defaults to 10.
        """
        content = self.get_page(link)
        processed = self.stemmer.process_text(content)
        query_tfidf = self.tfidf_query(processed)
        scores = self.scoring_methods[model](query_tfidf)

        self.show_recommendations(user, scores, top_n, scored_link=link)

        user.add_to_history(link, query_tfidf)

    def explain_similarity(self, link1, link2):
        """Hacky function to make dalex show breakdown plots of our cosine similarity

        Args:
            link1 (str): link to article 1
            link2 (str): link to article 2
        """
        tfidf1 = self.tfidf_query(self.stemmer.process_text(self.get_page(link1)))
        tfidf2 = self.tfidf_query(self.stemmer.process_text(self.get_page(link2)))

        elementwise = tfidf1 * tfidf2
        similarity = elementwise.sum() #cosine similarity is just a dot product due to normalization
        print(similarity)
        df = {
            "variable_name": self.tfidf.get_feature_names_out(),
            "variable_value":  elementwise,
            "variable": self.tfidf.get_feature_names_out(),
            "contribution": elementwise,
        }
        df = pd.DataFrame(df).sort_values(by="variable_value", ascending=False)
        df["cumulative"] = np.cumsum(df["variable_value"])
        df["sign"] = 1.0
        df['position'] = np.arange(1, df.shape[0] + 1)[::-1]
        df['label'] = "Cosine similarity"

        df_last = pd.DataFrame({"variable_name":'', "variable_value":'', "variable": "prediction", "contribution":similarity, 
                    "cumulative": 0, "sign": 0, "position": 0, "label": "Cosine similarity"}, index= [0] ,columns = df.columns)
        df_first = pd.DataFrame({"variable_name":"intercept","variable_value":'', "variable": "intercept", "contribution":0, 
                    "cumulative": 0, "sign": 0, "position": df.shape[0] + 1, "label": "Cosine similarity"}, index=[0], columns = df.columns)

        df = pd.concat([df_first, df, df_last]).reset_index(drop=True)

        bd = dx.predict_explanations._break_down.object.BreakDown()
        bd.result = df
        bd.plot()






