# Persuasion using sentential complement predicates

This repository contains data and code associated with the conference talk:
> Yiwei Luo. Taking sides using sentential complement predicates: The interplay of factivity and politeness in persuasion. 97th Annual Meeting of the Linguistic Society of America, January 7, 2023, Denver, CO.

Archival version coming soon!

## Getting started
1. Create and activate a Python 3.6 environment.
2. Run `pip install -r requirements.txt`.

## Repository contents
* `per_path_df.csv` is a dataframe with each row corresponding to an argument within the full dataset. Arguments are indexed by the root comment ID and leaf comment ID corresponding to the initial and final comments, repsectively, in the full argument subtree. The column `path_root_to_leaf` lists the full chain of comments comprising an argument, and `filtered_path_root_to_leaf` lists the chain of comments with deleted comments, empty comments, and comments written by the original user removed.
* `comment_ID2text.dill` is a dictionary with comment ID keys that can be used to look up the full and cleaned text corresponding to each comment 
* `predicate_lexicons.csv` contains lists of predicates categorized according to factive, concessive, etc. semantic categories
* `sci_ents.txt` contains the set of scientific entities (e.g., "study," "biologist") compiled using seed words and WordNet that was used to extract ngrams with a scientific source
* `main.ipynb` contains code for extracting features and fitting a logistic regression model to analyze persuasive argument features
* 
