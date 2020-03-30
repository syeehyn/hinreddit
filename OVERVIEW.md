# HinReddit


- [HinReddit](#hinreddit)
  - [1. Hateful Post Classification](#1-hateful-post-classification)
  - [2. Relation with HinDroid](#2-relation-with-hindroid)
  - [3. Related Works](#3-related-works)
  - [4. Techniques](#4-techniques)
      - [4.1. Data Ingestion](#41-data-ingestion)
      - [4.2. Labeling Training Data](#42-labeling-training-data)
      - [4.3. Classification Problem](#43-classification-problem)
  - [5. Weekly Schedule](#5-weekly-schedule)
  


## 1. Hateful Post Classification

As countless social platforms are developed and become accessible nowadays, the existence of hateful users also become more and more common on them, which motivates large numbers of scholars to apply various techniques in order to detect hateful users, or hateful speech.

In our project, we plan to investigate contents from Reddit, which is also a popular social network that focuses on aggregating American social news, rating web content and website discussion, that carries rich potential information of contents and their authors. Our goal is to classify hateful posts from the normal ones. Being able to identify hateful posts not only enables platforms to improve user experiences, but also helps to maintain a positive online environment.

Instead of natural language process techniques that are most commonly used in attempts to solve similar classification problems, we will be using graph embedding methods. Specifically, we will create a heterogeneous information network to capture the relationships among hateful posts and normal posts, which is then used as our features.

If our project is successful, we will have built an application, *hinReddit*, which helps identify hateful posts for Reddit. Similarly, others can apply our process on different social platforms. In addition, we will create a blog post including an EDA on the data we extracted, detailed description of the process we will complete to ingest data, perform feature engineering, and develop a neural network model, and finally a summary of the test result of our model.

## 2. Relation with HinDroid

Detecting hateful posts on Reddit is similar to our domain problem of detecting Android malware both conceptually and technically. Despite using different platforms, these two case studies both aim at identifying the malicious units from the benign units, and the goals are to produce a healthier and more positive environment to users. As we did in our replication using graph embedding techniques, here in our study, we will also pay attention to the connections as well as the communities of our object and construct heterogeneous information network (HIN) upon those connections that enables further training and classifications. 

Specifically, in our HIN graph, we will have Reddit post nodes equivalent to App nodes in the replication project and user-interaction nodes equivalent to API nodes in the replication. While Hindroid investigates more of the relationships among API calls, for instance, having three out of four matrices developing different interactions of APIs, and thus focuses less on relationships among Apps themselves, we plan to add to our HIN the relationship among Reddit post nodes themselves to further diversify our network graph. 

## 3. Related Works

Studies regarding the detection of hateful speech, content, and user in Online Social Networks have been manifold. In the report Characterizing and Detecting Hateful Users on Twitter, the authors present an approach to characterize and detect hate on Twitter at a user-level granularity. Their methodology consists of obtaining a generic sample of Twitter’s retweet graph, finding potential hateful users who employed words in a lexicon of hate-related words and running a diffusion process to sample more hateful users who are closely related in the neighborhood to those potential ones. However, there are still limitations to their approach. Their characterization has behavioral considerations of users only on Twitter, which lacks generality to be applied to other Online Social Networks platforms. Also, with ethical concerns, instead of labeling hate on a user-level, we want to avoid tagging individuals and believe that detecting hate on a content-level will be more impartial.


## 4. Techniques
The project has three main challenges: data ingestion, labeling training data for supervised learning, and finally our classification problem. 

#### 4.1. Data Ingestion

To obtain our dataset, we will make use of library PRAW (Python Reddit API Wrapper) to obtain Reddit post texts along with other information, such as its author, replies, reply authors, community/subreddit it belongs to, and its category. The challenge lies in the fact that the text data takes a considerably large space, so it is important for us to select the right subset of data that is available. Currently, we are planning to randomly sample a number of posts from a subset of subreddits in each category within the most recent year. The exact number is to be decided after we actually create data ingestion pipeline and learn the time/space resource we need.

#### 4.2. Labeling Training Data

After we obtain Reddit posts data, we will then divide it into training and test sets. Since we are going to perform supervised learning, labels for the training data are necessary. We plan to employ pretrained models, such as BERT and GPT-2, to label the training data.

#### 4.3. Classification Problem

Finally, with labeled training data In our project, we will use heterogeneous information network graph to depict relations among reddit posts and users, which are the two kinds of nodes in our HIN. We have proposed a few relations, that we may include in our HIN. The details of relations are shown below in Table 1. We will then apply node2vec to the HIN constructed, which provides us with the vector representation of each post. The result will become the input layer in our neural network to perform binary classification.

![relation table](relation_table.png)


## 5. Weekly Schedule

**Week 1:**

Goal: Create necessary functions for data ingestion

Tasks: 
* Create functions to obtain information of a Reddit post given post id/link. Information includes text content, author id, repliers’ ids, and category.
* Create functions to obtain information of a number of Reddit posts given subreddit name.
* Create functions to obtain information of a user given id, including friends’ ids and following subreddits.
* Create functions that clean data obtained to be ready for HIN construction
  

**Week 2:**

Goal: Finish data ingestion pipeline

Tasks:

* Create functions that randomly sample a number of subreddits given category name.
* Link functions created together to develop the complete data ingestion pipeline which given a category name and two numbers, ns and np, will first sample ns number of subreddits, then obtain data of np number of posts in the subreddit along with user data who at least reply to or create one of the sampled posts.
* Determine ideal values of ns and np that are used to obtain training and test sets.
* Start on using pretrained models to label training data


**Week 3:**

Goal: Start HIN construction

Tasks:
* Finish labeling training data if not yet finished
* Exploratory Data Analysis
* (Optional) Create a baseline model using extracted data directly as features
* Create functions to create a graph for each relation detailed above
  

**Week 4:**

Goal: Apply node2vec to turn HIN into vector representation

Tasks:
* Finish creating functions to create a graph for each relation
* Create a function to combine the graphs
* Finish creating the HIN given our training data
* Create functions that take in an HIN graph to create vectors for each post node 


**Week 5:**

Goal: Start constructing neural network

Tasks:
* Create functions that take in a new Reddit post and output its vector representation given our trained HIN
* Create functions that train a neural network with vectors we transformed from the HIN
* Start training vectors we obtain from our HIN


**Week 6:**

Goal: Finish testing and record result

Tasks:
* Finish training vectors if not yet finished
* Determine ideal metric
* Test on the testing dataset
* Compare result to that of the baseline model
* Discuss possible cases where pretrained model fail to identify hateful posts but our proposed methods do
* (Optional) Create Evasion Attack
  
**In each week, we will also record process and document related sections for the final blog post*
