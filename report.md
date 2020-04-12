# HinReddit

- [HinReddit](#HinReddit)
    - [1. Sentiment Analysis of Reddit Posts](#1-Sentiment-Analysis-of-Reddit-Posts)
    - [2. Relation with HinDroid](#2-relation-with-hindroid)
    - [3. Related Works](#3-related-works)
    - [4. Dataset](#4-Dataset)
    - [5. Data Ingestion Process](#5-Data-Ingestion-Process)
        - [1. Legal Issue](#51-Legal-Issue)
        - [2. Privacy Concerns](#52-Privacy-Concerns)
        - [3. Schema](#53-Schema)
        - [4. Pipeline](#54-Pipeline)
        - [5. Applicability of Pipeline](#55-Applicability-of-Pipeline)
    - [4. Graph Definitions & Computations](#4-Graph-Definitions-&-Computations)
    - [5. EDA on Apps](#5-EDA-on-Apps)
        - [5.1 EDA](#51-EDA)
        - [5.2 Reliability of Data](#52-Reliability-of-Data)
        - [5.3 Data Cleaning](#53-Data-Cleaning)
    - [6. Baseline Classification Model](#6-Baseline-Classification-Model)
    - [7. Kernel Construction](#7-Kernel-Construction)
        - [1. Hin Graph](#71-Hin-Graph)
        - [2. Kernel Classification Result](#72-Kernel-Classification-Result)
    - [8. Conclusion](#8-Conclusion)


## 1. Sentiment Analysis of Reddit Posts
As countless social platforms are developed and become accessible nowadays, more and more people get used to posting opinions on various topics online. These platforms thus become prolific sources for opinion mining, which motivates large numbers of scholars to apply various techniques in order to perform sentiment analysis. 

In our project, we plan to investigate contents from Reddit, which is also a popular social network that focuses on aggregating American social news, rating web content and website discussion, that carries rich potential information of contents and their authors. Our goal is to classify sentiment level of Reddit posts. The contents of these posts vary from personal opinions and interests to public discussions and statements. Being able to analyze posts sentimentally not only enriches platforms and organizations with feedbacks, but also helps to maintain a positive online environment by identifying the more negative posts and communities.

We plan to use Bidirectional Encoder Representations from Transformers (BERT), a neural network architecture transforming natural language processing (NLP) techniques, in our data ingestion pipeline for data labeling. However, instead of using NLP in attempts to solve classification problems, we will be using graph embedding methods. Specifically, we will create a heterogeneous information network to capture the relationships among Reddit posts, which is then used as our features.

If our project is successful, we will have built an application, *hinReddit*, which helps classify sentiment level of posts and identify more negative ones for Reddit. Similarly, others can apply our process on different social platforms. In addition, we will create a blog post including an EDA on the data we extracted, a map of subreddit communities by sentiment level, and detailed description of the process we will complete to ingest data. We will perform feature engineering, develop a neural network model, and finally a summary of the test result of our model.

## 2. Relation with HinDroid

Performing sentiment analysis on  Reddit posts is similar to our domain problem of detecting Android malware both conceptually and technically. Despite using different platforms, these two case studies both aim at grouping similar units from the entire population, with ours by sentiment level and the other by malicious level, and the goals are to produce a healthier and more positive environment to users by identifying the more negative units. As we did in our replication using graph embedding techniques, here in our study, we will also pay attention to the connections as well as the communities of our object and construct heterogeneous information network (HIN) upon those connections that enables further training and classifications. 

Specifically, in our HIN graph, we will have Reddit post nodes equivalent to App nodes in the replication project and user-interaction nodes equivalent to API nodes in the replication. While Hindroid investigates more of the relationships among API calls, for instance, having three out of four matrices developing different interactions of APIs, and thus focuses less on relationships among Apps themselves, we plan to add to our HIN the relationship among Reddit post nodes themselves to further diversify our network graph. 

## 3. Related Works

Studies regarding the detection of negative speech, content, and user in Online Social Networks have been manifold. In the report Characterizing and Detecting Hateful Users on Twitter, the authors present an approach to characterize and detect hate on Twitter at a user-level granularity. Their methodology consists of obtaining a generic sample of Twitter’s retweet graph, finding potential hateful users who employed words in a lexicon of hate-related words and running a diffusion process to sample more hateful users who are closely related in the neighborhood to those potential ones. However, there are still limitations to their approach. Their characterization has behavioral considerations of users only on Twitter, which lacks generality to be applied to other Online Social Networks platforms. Also, the definition of 'hateful' is vague and unclear in this specific work as well as many similar works. With our effort in reviewing relevent papers and websites, we find no unity nor clarity in defining the boundaries of 'hate.' Therefore, with ethical concerns, instead of labeling hate, we want to avoid tagging individuals or posts and believe that detecting attitude on a sentiment-level will be more impartial.
## 4. Dataset
To obtain our dataset, we use the API called [PushShift](https://github.com/pushshift/api) to obtain Reddit post information, including post text, title, and user ids who reply to either the post itself or any of the reply below the post and the comments that it provided. We use `PushShift` because it offers a specific API to obtain the flattened list of repliers' ids and takes considerably less time than doing the same with [PRAW](https://praw.readthedocs.io/en/latest/).

## 5. Data Ingestion Process
### 5.1 Legal Issue
As stated on the reddit's [terms of use](https://www.reddit.com/wiki/api-terms), in order to use the Reddit API we need to agree with all the terms listed on that website. After reading through the whole document, we understand that we have satisfied all the requirements. Morever, since we have register an account and fill out the form in agreeing with all the terms, we believe our usage of the Reddit API is legal. 
### 5.2 Privacy Concerns
As reddit.com is a public social platform, in which everyone can see the posts and comments people post, therefore we avoid violating data privacy. However, we can eliminate the posts or comments which touch with certain privacy issues. 
### 5.3 Schema
After extracting the posts and comments using the `PushShift` API, we have organized the data into three layers. As shown below, under the raw folder it contains the three layers, *post_detail*, *posts* and *comments*. The name of the files under each folder corresponds to each subrredit where the contents are taken from. 

```source
data/
|--raw/
|  |-- post_detail/
|  |   |-- science.json
|  |   |-- videos.json
|  |-- posts/
|  |-- |-- science.csv
|  |-- |-- videos.csv
|  |-- comments/
|  |   |-- science.csv
|  |   |-- videos.csv

```

- First Layer: Post detail 

The file contains certain number of posts id and all of its comments id under a certain subrredit. 
`submission_id` : id of the post <br />
`comment_ids`: id of each comment
```json
[{"submission_id":"fsoala","comment_ids":[]},{"submission_id": "fsnmj4", "comment_ids": ["fm2fd48", "fm2hrmh", "fm2k37i", "fm2k8p4", "fm2kuot", "fm2lces", "fm2lsao", "fm2lu4n", "fm2m5at", "fm3trkl", "fm4c7i6"]}]
```

- Second Layer: Posts <br />
The csv file contains the information of each post in a dateframe where the unit of observation is the individual post. <br>
`id`: post_id <br />
`author`: username of the author who make the post <br>
`title`: title of the post <br>
`selftext`
`num_comments`: number of comments <br>
`created_utc`: the epoch date for which the post is created <br>
`full_link`: the link to the reddit post <br>
`subreddit`: subreddit it belongs to <br>
`score`: number of upvote - number of downvote

- Third Layer: Comments <br />
The csv file contains the information of each specific post in a dataframe where the unit of observation is the individual comment. <br>
`id`: comment id <br>
`author` : username of the author who make the comment <br>
`created_utc` : the epoch date for which the comment is made <br>
`is_submitter`: whether that person post the original post
`subreddit`: the subreddit it belongs to <br>
`link_id`: the post id for which this comment is made for
`send_replies`

### 5.4 Pipeline
- Create `config/data-params.json`, an example shown below. Information includes: POST_ARGS: parameter related to the post extraction part. META_ARGS: parameter related to the comment extraction part. The all the posts is sorted by the creation data and we extracted data prior to the date of `Tuesday,March 31 17:00:00 2020 PDT`.
```json
{"POST_ARGS":
    {"sort_type":"created_utc",
    "sort":"dsc",
    "size":"1000",
    "start":"1585699200"},
"META_ARGS":
    {"filepath":".\/tests",
    "total":"1000",
    "meta":["id","author","title","selftext","num_comments","created_utc","full_link","subreddit","score"],
    "subreddits":["amitheasshole","showerthoughts","politics","documentaries"]}}
```