**Week 1:**

3.29 - 4.4

Goal: Create necessary functions for data ingestion

Tasks: 

* Create functions to obtain information of a Reddit post given post id/link. Information includes text content, author id, repliers’ ids, and category.
* Create functions to obtain information of a number of Reddit posts given subreddit name.
* Create functions to obtain information of a user given id, including friends’ ids and following subreddits.
* Create functions that clean data obtained to be ready for HIN construction
  

**Week 2:**

4.5 - 4.11

Goal: Finish data ingestion pipeline

Tasks:

* Create functions that randomly sample a number of subreddits given category name.
* Link functions created together to develop the complete data ingestion pipeline which given a category name and two numbers, ns and np, will first sample ns number of subreddits, then obtain data of np number of posts in the subreddit along with user data who at least reply to or create one of the sampled posts.
* Determine ideal values of ns and np that are used to obtain training and test sets.
* Start on using pretrained models to label training data


**Week 3:**

4.12 - 4.18

Goal: Start HIN construction

Tasks:
* Finish labeling training data if not yet finished
* Exploratory Data Analysis
* (Optional) Create a baseline model using extracted data directly as features
* Create functions to create a graph for each relation detailed above
  

**Week 4:**

4.19 - 4.25

Goal: Apply node2vec to turn HIN into vector representation

Tasks:
* Finish creating functions to create a graph for each relation
* Create a function to combine the graphs
* Finish creating the HIN given our training data
* Create functions that take in an HIN graph to create vectors for each post node 


**Week 5:**

4.26 - 5.2

Goal: Start constructing neural network

Tasks:
* Create functions that take in a new Reddit post and output its vector representation given our trained HIN
* Create functions that train a neural network with vectors we transformed from the HIN
* Start training vectors we obtain from our HIN


**Week 6:**

5.3 - 5.9

Goal: Finish testing and record result

Tasks:
* Finish training vectors if not yet finished
* Determine ideal metric
* Test on the testing dataset
* Compare result to that of the baseline model
* Discuss possible cases where pretrained model fail to identify hateful posts but our proposed methods do
* (Optional) Create Evasion Attack
  
**In each week, we will also record process and document related sections for the final blog post*