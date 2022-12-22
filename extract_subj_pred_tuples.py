#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_text_chunks', type=str,
                    help='where to read in chunked argument texts from')
parser.add_argument('--path_to_arguments', type=str,
                    help='where to read in dataframe of arguments from')
parser.add_argument('--output_fname', type=str,
                    help='where to save extract subject-predicate tuples')
args = parser.parse_args()

def flatten(l):
    return [item for sublist in l for item in sublist]

def spacy_preprocess(path_to_text_chunks, batch_size=100):
    import spacy
    from spacy import displacy
    from spacy.tokens import DocBin

    nlp = spacy.load("en_core_web_sm")

    all_chunks_df = pd.read_csv(path_to_text_chunks,sep='\t')
    all_chunks_df['batch_no'] = pd.Series(all_chunks_df.index).floordiv(batch_size)
    all_chunks_df['index_within_batch'] = pd.Series(all_chunks_df.index) % batch_size
    all_chunks_df['batch_loc'] = all_chunks_df.apply(lambda x: f"{x['batch_no']}|{x['index_within_batch']}", axis=1)
    batch_loc2all_chunks_ix = dict(zip(all_chunks_df['batch_loc'],all_chunks_df.index))
    print(all_chunks_df.shape)

    # Iterate over batches and serialize along the way

    if not os.path.exists('pickled_spacy_docs'):
        os.mkdir('pickled_spacy_docs')

    for batch_start_ix in trange(len(all_chunks_df),batch_size):
        batch_no = int(batch_start_ix/batch_size)
        batch_end_ix = batch_start_ix+batch_size
        #print(f"Indices for batch no. {batch_no}: ({batch_start_ix}, {batch_end_ix})")
        texts = all_chunks_df['text'][batch_start_ix:batch_end_ix].values
        docs = []
        doc_bin = DocBin(attrs=["ORTH", "TAG", "HEAD", "DEP", "LEMMA", "POS", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID"],
                         store_user_data=True)
        for doc in nlp.pipe(texts):#, disable=["tok2vec"]):
            docs.append(doc)
            doc_bin.add(doc)
        bytes_data = doc_bin.to_bytes()
        pickle.dump(bytes_data,open(f'pickled_spacy_docs/batch_{batch_no}_bytes_data.pkl','wb'))
        
def extract_quote_indices(doc, doc_ix, verbose=False):
    
    #print('HI!')
    punct_marks = [t.text for t in doc if t.pos_ == 'PUNCT']
    #print(punct_marks)
    counted_punct_marks = Counter(punct_marks)
    
    if verbose:
        print(counted_punct_marks)
    
    if (counted_punct_marks['“'] != counted_punct_marks['”']) \
    or (counted_punct_marks['"'] % 2 != 0) \
    or ((counted_punct_marks['“'] > 0) and (punct_marks.index('”') < punct_marks.index('“'))):
        #print(f"Quote mismatch in doc no. {doc_ix}; skipping.")
        return []
    else:
        stack, quotes = [], []
        for step, curr_tok in enumerate(doc):
            if verbose:
                print(f"Step: {step}\t Token: {curr_tok.text}\t stack: {stack}\t quotes: {quotes}")
            if curr_tok.text == '“': # definitely a left quote
                stack.append((curr_tok.text, curr_tok.i))
            elif curr_tok.text == '”': # definitely a right quote
                try: 
                    assert (stack[-1][0] == '“')
                    quotes.append((stack[-1][1], curr_tok.i))
                    stack.pop()
                except (AssertionError, IndexError) as e:
                    print("On doc:", doc_ix)

            elif curr_tok.text == '"': # could be left or right quote
                if (len(stack) == 0) or (stack[-1][0] != '"'): # if no prior such quotes, then left quote 
                    stack.append((curr_tok.text, curr_tok.i))
                elif stack[-1][0] == '"': # if there is a prior such quote, then right quote
                    quotes.append((stack[-1][1], curr_tok.i))
                    stack.pop()
                else: 
                    pass
            else:
                pass

        return quotes

def extract_subj_pred_tups(doc, doc_ix, 
                           pred_set=preds_set, mwe_pred_set=mwe_preds_set, neg_lemma_set={'no','not',"n't",'none'},
                           verbose=False):
    
    if verbose:
        print("Doc ix:", doc_ix)
    
    # Step 1- Find potential SCPs/MVPs belonging to lexicon
    preds = []
    for tok in doc:
        if (tok.pos_ == 'VERB') and \
        ((tok.head.i == tok.i) or # is root   
         (tok.dep_ == 'csubj') or # is copular subject
         ((tok.dep_ == 'conj') and (tok.head.head.i == tok.head.i)) or # is conjoined to root
         ((tok.dep_ == 'xcomp') and (tok.head.dep_ == 'root'))): # is xcomp of root
            # token is ROOT or complement subject or conjunct of ROOT
            tok_particles = [c for c in tok.rights if c.dep_ in {'prt','prep','compound:prt'}]
            if verbose:
                print(f"Found the following particles accompanying main verb, {tok.text, tok.lemma_}:", tok_particles)
            if len(tok_particles) > 1:
#                 print("Multiple particles found!")
#                 print(tok_particles)
                tok_prt = tok_particles[0]
                if verbose:
                    print(f"\tChecking whether '{tok.lemma_} {tok_prt.lemma_}' is among MWE preds...")
                    print('\t\t'+str(f'{tok.lemma_} {tok_prt.lemma_}' in mwe_pred_set))
                if f'{tok.lemma_} {tok_prt.lemma_}' in mwe_pred_set:
                    preds.append([tok, tok_prt])
            elif len(tok_particles) == 1: # check if it's among the MWE preds
                tok_prt = tok_particles[0]
                if verbose:
                    print(f"\tChecking whether '{tok.lemma_} {tok_prt.lemma_}' is among MWE preds...")
                    print('\t\t'+str(f'{tok.lemma_} {tok_prt.lemma_}' in mwe_pred_set))
                if f'{tok.lemma_} {tok_prt.lemma_}' in mwe_pred_set:
                    preds.append([tok, tok_prt])
            else: # check if it's among the lexicon
                if tok.lemma_ in pred_set:
                    if verbose:
                        print(f"\tFound potential SWE pred `{tok.lemma_}` in lexicon.")
                    # if it's among polysemous preds, check that it embeds a CC
                    if tok.lemma_ in PREDS_REQUIRING_CC:
                        if verbose:
                            print(f"\t\tFound polysemous predicate; checking for ccomp...")
                        if 'ccomp' in set([c.dep_ for c in tok.children]):
                            preds.append([tok])
                            if verbose:
                                print("\t\t\tccomp found!")
                    else:
                        preds.append([tok])
    if verbose:
        print("Candidate SCPs:")
        print([[tok.text for tok in pred] for pred in preds])
                    
    # Step 1b- Get full SCP tokens
    pred_mods = {}
    for pred in preds:
        main_pred = pred[0]
        if pred[0].dep_ == 'root':
            root_pred = pred[0] 
            full_pred_toks = []
        else:
            root_pred = pred[0].head
            full_pred_toks = [root_pred]
        #assert root_pred.dep_ == 'root'
        full_pred_toks.extend([c for c in root_pred.children 
                               if (c.dep_ not in {'ccomp','nsubj','nsubjpass'})
                               and (c != main_pred)])
        #print('Valid pred children:', full_pred_toks)
        grandchildren = flatten([c.children for c in full_pred_toks])
        while len(grandchildren) > 0:
            full_pred_toks.extend(grandchildren)
            grandchildren = flatten([g.children for g in grandchildren])
        #full_pred_toks.extend(pred)
        in_order_pred_toks = sorted(full_pred_toks, key=lambda x: x.i)
        pred_mods[main_pred.i] = in_order_pred_toks
    if verbose:
        print('\nModification:')
        print({key: [x.text for x in pred_mods[key]]
               for key in pred_mods})
    
    # Step 2- Check if candidate preds are negated
    neg_modifiers_per_pred = {} # indexed by the main_pred's index in doc
    for pred in preds:
        main_pred = pred[0]
        root_pred = pred[0] if pred[0].dep_ == 'root' else pred[0].head
        #assert root_pred.dep_ == 'root'
        neg_modifiers = [c for c in root_pred.children
                         if (c.dep_ == 'neg') 
                         or ((c.dep_ in {'det','advmod'}) and (c.lemma_ in neg_lemma_set))]
        neg_modifiers_per_pred[main_pred.i] = neg_modifiers
        # TO DO: handle polarity-reversing adverbial modifiers, e.g.
        # almost, hardly, not at all, under no circumstances
    if verbose:
        print('\nNegation:')
        print({key: [x.text for x in neg_modifiers_per_pred[key]]
               for key in neg_modifiers_per_pred})
                         
    # Step 3- Check if candidate preds have modal modification
    modals_per_pred = {}
    for pred in preds:
        main_pred = pred[0]
        if pred[0].dep_ == 'root':
            root_pred = pred[0] 
            modals = []
        else:
            root_pred = pred[0].head
            modals = [root_pred]
        #assert root_pred.dep_ == 'root'
        modals.extend([c for c in root_pred.children
                       if (c.dep_ == 'aux')]) # handle cases to exclude based on lemmas post-hoc
        in_order_modals = sorted(modals, key=lambda x: x.i)
        modals_per_pred[main_pred.i] = in_order_modals
    if verbose:
        print('\nAuxiliaries:')
        print({key: [x.text for x in modals_per_pred[key]]
           for key in modals_per_pred})
    
    # Step 4- Check if candidate preds are part of a question
    questions_per_pred = {}
    for pred in preds:
        main_pred = pred[0]
        root_pred = pred[0] if pred[0].dep_ == 'root' else pred[0].head
        #assert root_pred.dep_ == 'root'
        if main_pred.dep_ == 'csubj':
            questions = [c for c in main_pred.head.children
                         if (c.dep_ == 'punct') and (c.lemma_ == '?')]
        else:
            questions = [c for c in root_pred.children
                         if (c.dep_ == 'punct') and (c.lemma_ == '?')]
        questions_per_pred[main_pred.i] = questions
    if verbose:
        print('\nQuestioning:')
        print({key: [x.text for x in questions_per_pred[key]]
           for key in questions_per_pred})
    
    # Step 5- Find subjects of SCPs
    subjs_per_pred = {} # indexed by predicate index
    for pred in preds:
        main_pred = pred[0]
        root_pred = pred[0] if pred[0].dep_ == 'root' else pred[0].head
        #assert root_pred.dep_ == 'root'
        subjs = [c for c in root_pred.children if c.dep_ in {'nsubj','csubj'}]
        passive_subjs = [c for c in root_pred.children if c.dep_ == 'nsubjpass']
        
        if len(subjs) > 1:
#             print("Multiple subjects found!")
#             print(subjs)
#             print([tok.text for tok in doc])
            subjs_per_pred[main_pred.i] = subjs[0]
        elif len(subjs) == 1:
            subjs_per_pred[main_pred.i] = subjs[0]
        else: # case of passive, ellipted, pro-drop, or imperative; if not first 2, then assume pro-drop
            if len(passive_subjs) > 0:
                subjs_per_pred[main_pred.i] = 'passive'
            else:
                if main_pred.dep_ == 'conj': # check if it's conjoined to something w/ a subject
                    conjunct = main_pred.head
                    if conjunct.i in subjs_per_pred:
                        subjs_per_pred[main_pred.i] = subjs_per_pred[conjunct.i]
                    else:
                        subjs_per_pred[main_pred.i] = f'replace with conjunct index={conjunct.i} subj'
                else:
                    subjs_per_pred[main_pred.i] = 'pro-drop'
    if verbose:
        print('\nSubjects:')
        print({key: subjs_per_pred[key].text 
               if type(subjs_per_pred[key]) != str else subjs_per_pred[key]
               for key in subjs_per_pred})
            
    # Step 6- Aggregate information
    out = defaultdict(lambda: defaultdict(dict))
    for pred in preds:
        main_pred = pred[0]
        subj_tok = subjs_per_pred[main_pred.i]
        if (type(subj_tok) == str) and (subj_tok == 'pro-drop'): # pro-drop case
            out[main_pred.i]['main_subj']['index'] = None
            out[main_pred.i]['main_subj']['text'] = '[PRO-DROP]'
            out[main_pred.i]['main_subj']['lemma'] = 'i'
            out[main_pred.i]['subj_mods']['text'] = []
            out[main_pred.i]['subj_mods']['lemma'] = []
            out[main_pred.i]['subj_mods']['dep'] = []
        elif (type(subj_tok) == str) and (subj_tok == 'passive'): # passive
            out[main_pred.i]['main_subj']['index'] = None
            out[main_pred.i]['main_subj']['text'] = '[PASSIVE]'
            out[main_pred.i]['main_subj']['lemma'] = '[PASSIVE]'
            out[main_pred.i]['subj_mods']['text'] = []
            out[main_pred.i]['subj_mods']['lemma'] = []
            out[main_pred.i]['subj_mods']['dep'] = []
        elif (type(subj_tok) == str) and (subj_tok[:7]=='replace'): # conjoined subject
            conjunct_index = int(subj_tok.split('=')[-1].split(' ')[0])
            conjunct_subj = [tok for tok in doc if (tok.dep_[:5] == 'nsubj') and (tok.head.i == conjunct_index)]
            if len(conjunct_subj) > 0:
                conjunct_subj = conjunct_subj[0]
                out[main_pred.i]['main_subj']['index'] = 'ellipted'
                out[main_pred.i]['main_subj']['text'] = conjunct_subj.text
                out[main_pred.i]['main_subj']['lemma'] = conjunct_subj.lemma_
                out[main_pred.i]['subj_mods']['text'] = [c.text for c in conjunct_subj.subtree
                                                         if c != conjunct_subj]
                out[main_pred.i]['subj_mods']['lemma'] = [c.lemma_ for c in conjunct_subj.subtree
                                                          if c != conjunct_subj]
                out[main_pred.i]['subj_mods']['dep'] = [c.dep_ for c in conjunct_subj.subtree
                                                        if c != conjunct_subj]
            else:
                out[main_pred.i]['main_subj']['index'] = None
                out[main_pred.i]['main_subj']['text'] = None
                out[main_pred.i]['main_subj']['lemma'] = None
                out[main_pred.i]['subj_mods']['text'] = None
                out[main_pred.i]['subj_mods']['lemma'] = None
                out[main_pred.i]['subj_mods']['dep'] = None
        else:
            out[main_pred.i]['main_subj']['index'] = subj_tok.i
            out[main_pred.i]['main_subj']['text'] = subj_tok.text
            out[main_pred.i]['main_subj']['lemma'] = subj_tok.lemma_
            out[main_pred.i]['subj_mods']['text'] = [c.text for c in subj_tok.subtree
                                                if c != subj_tok]
            out[main_pred.i]['subj_mods']['lemma'] = [c.lemma_ for c in subj_tok.subtree
                                                 if c != subj_tok]
            out[main_pred.i]['subj_mods']['dep'] = [c.dep_ for c in subj_tok.subtree
                                               if c != subj_tok]
        out[main_pred.i]['main_pred']['index'] = [p.i for p in pred]
        out[main_pred.i]['main_pred']['text'] = [p.text for p in pred]
        out[main_pred.i]['main_pred']['lemma'] = [p.lemma_ for p in pred]
        out[main_pred.i]['pred_mods']['text'] = [t.text for t in pred_mods[main_pred.i]]
        out[main_pred.i]['pred_mods']['lemma'] = [t.lemma_ for t in pred_mods[main_pred.i]]
        out[main_pred.i]['pred_mods']['dep'] = [t.dep_ for t in pred_mods[main_pred.i]]
        out[main_pred.i]['pred_neg']['text'] = [x.text for x in neg_modifiers_per_pred[main_pred.i]]
        out[main_pred.i]['pred_neg']['lemma'] = [x.lemma_ for x in neg_modifiers_per_pred[main_pred.i]]
        out[main_pred.i]['pred_modal']['text'] = [x.text for x in modals_per_pred[main_pred.i]]
        out[main_pred.i]['pred_modal']['lemma'] = [x.lemma_ for x in modals_per_pred[main_pred.i]]
        out[main_pred.i]['pred_q']['text'] = [x.text for x in questions_per_pred[main_pred.i]]
        out[main_pred.i]['pred_q']['lemma'] = [x.lemma_ for x in questions_per_pred[main_pred.i]]
        out[main_pred.i]['pred_cond']['text'] = [x.text for x in main_pred.children
                                                 if x.lemma_.lower() == 'if']
        out[main_pred.i]['pred_cond']['lemma'] = [x.text for x in main_pred.children
                                                  if x.lemma_.lower() == 'if']
        
    # Step 7- Annotate for whether tuples occur in quotes
    quotes = extract_quote_indices(doc, doc_ix)
    if verbose:
        print('\nQuote indices:', quotes)
    if len(quotes) > 0:
        for pred_ix_key in out:
            tup_ixs = []
            tup = []
            subj_i = out[pred_ix_key]['main_subj']['index']
            if (subj_i is not None) and (subj_i != 'ellipted'):
                tup_ixs.append(subj_i)
                tup.append(out[pred_ix_key]['main_subj']['text'])
            pred_ixs = out[pred_ix_key]['main_pred']['index']
            tup_ixs.extend(pred_ixs)
            tup.extend(out[pred_ix_key]['main_pred']['text'])
            tup_start, tup_end = min(tup_ixs), max(tup_ixs)
            for quote_start, quote_end in quotes:
                if (int(tup_start) > int(quote_start)) and (int(tup_end) < int(quote_end)):
                    out[pred_ix_key]['in_quote'] = True
                    if verbose:
                        print('\tFound tuple within quote:', tup_ixs, tup)
                    break
                else:
                    out[pred_ix_key]['in_quote'] = False
    else:
        for pred_ix_key in out:
            out[pred_ix_key]['in_quote'] = False
    
    return out
        
def subj_is_negated(subj_pred_tup, neg_lemmas={'no','not','nothing','never','nowhere',}):
    if len(subj_pred_tup['subj_neg']) > 0:
        return True
    else:
        full_subj_lemmas = set([x[1] for x in subj_pred_tup['full_subj']])
        if len(full_subj_lemmas.intersection(neg_lemmas)) > 0:
            return True
        else:
            return False
        
def pred_is_negated(subj_pred_tup, neg_lemmas={'no','not','nothing','never','nowhere',}):
    if len(subj_pred_tup['pred_neg']) > 0:
        return True
    else:
        full_pred_lemmas = set([x[1] for x in subj_pred_tup['full_pred']])
        if len(full_pred_lemmas.intersection(neg_lemmas)) > 0:
            return True
        else:
            return False
    
def pred_is_modalized(subj_pred_tup, modal_lemmas={'should','ought','must'}):#might, would, may):
    if len(set([x[1] for x in subj_pred_tup['pred_aux']]).intersection(modal_lemmas)) > 0:
        return True
    return False

def pred_is_questioned(subj_pred_tup, q_lemmas={'?'}):
    if len(set([x[1] for x in subj_pred_tup['pred_children']]).intersection(q_lemmas)) > 0:
        return True
    return False

def pretty_report_subj_pred_tups(subj_pred_tups):
    """
    | subj negation | main_subject text (lemma, pos, full) | pred modal | pred negation | main pred text (lemma, pos, full)| 
    """
    d = defaultdict(list)
    for key in subj_pred_tups:
        tup_ = subj_pred_tups[key]
        d['subj. negation'].append(subj_is_negated(tup_))
        d['main subj.'].append(
            f"{tup_['main_subj']['text']} ({tup_['main_subj']['lemma']}, {tup_['main_subj']['pos']}|\
{' '.join([x[0] for x in tup_['full_subj']])})")
        d['pred. has modal'].append(pred_is_modalized(tup_))
        d['pred. has negation'].append(pred_is_negated(tup_))
        d['main pred.'].append(
            f"{tup_['main_pred']['text']} ({tup_['main_pred']['lemma']}, {tup_['main_pred']['pos']}|\
{' '.join([x[0] for x in tup_['full_pred']])})")
        d['is question'].append(pred_is_questioned(tup_))
    
    return pd.DataFrame(d)

def batch_extract_subj_pred_tups(docs_batch, verbose=False):
    return {ix_doc: extract_subj_pred_tups(doc, ix_doc, verbose=verbose) for ix_doc, doc in enumerate(docs_batch)}

def main(path_to_text_chunks, path_to_arguments, output_fname):
    spacy_preprocess(path_to_text_chunks)
    
    per_path_df = pd.read_csv(path_to_arguments)
    for col in ['path_root_to_leaf','filtered_path_root_to_leaf']:
        per_path_df[col] = per_path_df[col].apply(lambda x: json.loads(x))
    subtree_guid2outcome = dict(zip(per_path_df['guid'], per_path_df['won_delta']))
    subtree_guid2path = dict(zip(per_path_df['guid'], per_path_df['filtered_path_root_to_leaf']))
 
    utt_id2batch_locs = defaultdict(list)
    for _,row in tqdm(all_chunks_df.iterrows()):
        utt_id2batch_locs[row['utt_id']].append(row['batch_loc'])

    def get_subtree_tups(subtree_guid):
        subtree_path = subtree_guid2path[subtree_guid]
        subtree_batch_locs = flatten([utt_id2batch_locs[utt_id] for utt_id in subtree_path])
        subtree_tups = flatten([batch_loc2tups[batch_loc] for batch_loc in subtree_batch_locs
                        if batch_loc in batch_loc2tups])
        return subtree_tups
   
    subj_pred_tups_per_subtree = defaultdict(list)
    for _,row in tqdm(per_path_df.iterrows()):
        guid = row['guid']
        _tups = get_subtree_tups(guid)
        subj_pred_tups_per_subtree[guid].extend(_tups)

    if not os.path.exists('output'):
        os.mkdir('output')
    dill.dump(subj_pred_tups_per_subtree,open(os.path.join('output',output_fname),'wb'))
    
if __name__ == "__main__":
    main(args.path_to_text_chunks, args.path_to_arguments, args.output_fname)