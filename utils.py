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
from nltk.tokenize import word_tokenize, wordpunct_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


porter = PorterStemmer()
lancaster = LancasterStemmer()
wordnet = WordNetLemmatizer()


class CustomStemmer():
    def __init__(self, data, stemmer=LancasterStemmer, tokenizer=word_tokenize, lemmatizer=WordNetLemmatizer, custom_stopwords=stopwords.words('english')):
        self.data = data
        self.stemmer = stemmer()
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer()
        self.custom_stopwords = custom_stopwords

    def process_text(self, text, stem=False, lemmatize=True):
        words = self.tokenizer(text)
        final_words = []  # can't use set to preserve order
        for word in words:
            if re.match(r'^[A-Za-z]*$', word):
                final_words.append(word)
        cleaned = [
            word for word in final_words if word not in self.custom_stopwords]

        text = ' '.join(cleaned)
        if stem:
            text = self.stemmer.stem(text)
        elif lemmatize:
            text = self.lemmatizer.lemmatize(text)

        return text

    def process_corpus(self):
        self.processed = {index: self.process_text(
            element['content']) for index, element in self.data.iterrows()}
        return self.processed

    def generate_csv(self):
        df = pd.DataFrame(self.processed, index=[0]).T
        df.columns = ['text']
        df.to_csv('processed.csv', columns=['text'])


class BFSScraper():
    def __init__(self, n_to_visit):
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
        links = parsed_page.find_all(
            'a', attrs={'href': re.compile(r'^\/wiki\/(?!File)(?!Main_Page)\w*$')})  # To get only wikipedia articles, doen't take files nor something with :, ( etc. Don't go back to main page

        # To remove duplicates, probably not the most efficient way
        links = list(set(links))

        return links

    def process_one_link(self, link):
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

    def dummy_generator(self, n):
        while len(self.pages) < n:
            yield

    def generate_summary(self):
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
        # with open("text.csv" , 'w', encoding="utf-8") as f:
        #     for link, page in self.pages.items():
        #         f.write(f"{link}, {page['content'].strip()}\n")

    def bfs(self, starting_link, n=1000):
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


def generator(texts):
    while len(texts) < 1500:
        yield


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


class Ranker:
    def __init__(self, data):
        self.docs = data
        self.docs = data['text']

        self.init_tfidf()
        self.init_svd()
        self.init_pca()

    def init_tfidf(self):
        self.tfidf = TfidfVectorizer(use_idf=True, smooth_idf=False)
        trans = self.tfidf.fit_transform(self.docs)
        self.trans_np = trans.toarray()
        self.dfTFIDF = pd.DataFrame(self.trans_np, index=np.arange(
            len(self.docs)), columns=self.tfidf.get_feature_names_out())

    def init_svd(self):
        to_svc = self.trans_np.T
        u, s, vh = np.linalg.svd(to_svc, full_matrices=False)

        u_c = np.concatenate(u[:, None])
        vh_c = np.concatenate(vh[:, None])
        s_shape = (s.shape[0], s.shape[0])
        s_c = np.zeros(s_shape)
        s_c[np.arange(s_shape[0]), np.arange(
            s_shape[0])] = s
        s_c_inv = np.zeros(s_shape)
        s_c_inv[np.arange(s_shape[0]), np.arange(
            s_shape[0])] = 1/s

        reconstruction = u_c @ s_c @ vh_c

        assert np.allclose(to_svc, self.reconstruction) == True

        how_many = (s_c > 1).sum()
        self.u_c_k = u_c[:, :how_many]
        self.s_c_inv_k = self.s_c_inv[:how_many, :how_many]
        vh_c_k = self.vh_c[:how_many, :]

        self.dfSVD = pd.DataFrame(
            vh_c_k.T, index=np.arange(len(self.docs)))

    def init_pca(self, printing=False):
        self.pca = PCA(n_components=1000)
        self.pca.fit(self.trans_np)
        if printing:
            print(self.pca.explained_variance_ratio_.sum())
        docs_transformed = self.pca.transform(self.trans_np)
        self.dfPCA = pd.DataFrame(
            docs_transformed, index=np.arange(len(self.docs)))

    def tfidf(self, query):
        query = self.tfidf.transform([query]).toarray()[0]
        return 1-self.dfTFIDF.apply(lambda x: cosine(x, query), axis=1)

    def pca(self, query):
        query = self.tfidf.transform([query]).toarray()[0]
        query_pca = self.pca.transform(query[None, :])
        return 1-self.dfPCA.apply(lambda x: cosine(x, query_pca), axis=1)

    def svd(self, query):
        query = self.tfidf.transform([query]).toarray()[0]
        query_svd = query @ self.u_c_k @ self.s_c_inv_k
        return 1-self.dfSVD.apply(lambda x: cosine(x, query_svd), axis=1)

    def rank(self, prev_docs, model='tfidf'):
        ranks = []
        if model == 'tfidf':
            ranks = [self.tfidf(doc) for doc in prev_docs]
        elif model == 'pca':
            ranks = [self.pca(doc) for doc in prev_docs]
        elif model == 'svd':
            ranks = [self.svd(doc) for doc in prev_docs]
        else:
            raise TypeError("pca, svd and tfidf are the only options!")
        return np.array(ranks).mean()
