# WikipediaRecommender

This is a Wikipedia Pages Recommender created for Information Retrieval classes. 

## Details
It uses motified BFS to go through a graph of links starting from a given page, and adding to the queue randomly next `n` pages, to which the current page links. After collecting the pages, we check the database statistics with some plots. We also check the Zipf's and Heaps' laws. Next, we calculate the documents' similarity using TF-IDF, PCA or SVD. The option being used is user-dependant. At the end we propose articles to read given previous pages. 