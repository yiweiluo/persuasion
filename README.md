# Persuasion using sentential complement predicates

This repository contains data and code associated with the conference talk:
> Yiwei Luo. Taking sides using sentential complement predicates: The interplay of factivity and politeness in persuasion. 97th Annual Meeting of the Linguistic Society of America, January 7, Denver, CO.

Archival version coming soon!

## Getting started
1. Create and activate a Python 3.6 environment.
2. Run `pip install -r requirements.txt`.
3. Update the `config.json` file with your local OS variables and a Reddit API key, if you would like to gather custom data from r/ChangeMyView, rather than use the cached data files provided.

## Repository structure
### Data
* `predicate_lexicons.csv` contains lists of predicates categorized according to factive, concessive, etc. semantic categories
* `sci_ents.txt` contains the set of scientific entities (e.g., ``study,'' ``biologist'') compiled using seed words and WordNet that was used to extract ngrams with a scientific source
### Code
* 
