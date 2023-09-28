import os.path

from transformers import AutoConfig, AutoTokenizer
import torch
import sys
import jsonlines
from tqdm.auto import tqdm
sys.path.insert(0, '.')
from copy import copy
from more_itertools import flatten
from unidecode import unidecode
import re
from util import label_mapper
import numpy as np
import pandas as pd
import sqlite3
from more_itertools import unique_everseen
tqdm.pandas()

MAX_DOC_LENGTH = 8000
MAX_SENT_LENGTH = 300
MAX_NUM_SENTS = 100
from nltk.corpus import stopwords
try:
    ENGLISH_STOPWORDS = stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    ENGLISH_STOPWORDS = stopwords.words('english')

PUNCTUATION_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~' # except "'"
SENT_LEN_TO_EXCLUDE = 4


def filter_stopwords(x):
    x_copy = unidecode(copy(x).lower())
    for p in PUNCTUATION_TO_REMOVE:
        x_copy = x_copy.replace(p, ' ')
    for word in ENGLISH_STOPWORDS:
        x_copy = re.sub(r'\b%s\b' % word, ' ', x_copy, flags=re.IGNORECASE)
    x_copy = re.sub('\s+', ' ', x_copy)
    return x_copy.strip()


def is_acceptable_sentence(x):
    x = filter_stopwords(x.strip())
    return len(x.split()) > SENT_LEN_TO_EXCLUDE


def get_tokenizer_name(model_name, tokenizer_name):
    if tokenizer_name is not None:
        return tokenizer_name
    if 'roberta-base' in model_name:
        return 'roberta-base'
    elif 'roberta-large' in model_name:
        return 'roberta-large'


def get_model_and_dataset_class(model_type):
    if model_type == 'sentence':
        from sentence_model import SentenceClassificationModel as model_class
        from sentence_model import TokenizedDataset as dataset_class
    else:
        from full_sequence_model import LongRangeClassificationModel as model_class
        from full_sequence_model import TokenizedDataset as dataset_class
    return model_class, dataset_class


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_data(args, tokenizer):
    def _read_data(args):
        if '.csv' in args.dataset_name:
            print('reading csv...')
            df = pd.read_csv(args.dataset_name)
        elif '.jsonl' in args.dataset_name:
            print('reading json...')
            df = pd.read_json(args.dataset_name, lines=True)
        elif '.db' in args.dataset_name:
            print('reading from db...')
            conn = sqlite3.connect(args.dataset_name)
            df = pd.read_sql(args.sql_command, conn)
        return df

    input_df = _read_data(args)
    input_df = (
        input_df
            .loc[lambda df: df[args.text_col].notnull()]
            .drop_duplicates(args.text_col)
            .loc[lambda df: df[args.text_col].str.len() < MAX_DOC_LENGTH * 2]
    )
    if args.n_rows is not None:
        # debug
        input_df = input_df.iloc[:args.n_rows]

    if 'sentences' not in input_df.columns:
        dirname, basename = os.path.split(args.dataset_name)
        sentences_fn = os.path.join(dirname, basename.split('.')[0] + '-sentences.jsonl')
        if os.path.exists(sentences_fn):
            print('loading sentences from file...')
            sentenes_df = pd.read_json(sentences_fn, lines=True)[[args.id_col, 'text_short', 'sentences']]
            input_df = input_df.merge(sentenes_df, on=args.id_col, how='left')
        else:
            input_df['sentences'] = np.nan

        if 'text_short' not in input_df.columns:
            print('splitting by max_token_length...')
            input_df['text_short'] = (
                input_df[args.text_col]
                .progress_apply(lambda x: tokenizer.encode(x)[:MAX_DOC_LENGTH])
                .progress_apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
            )

        input_df = sentencize_col(input_df, args, tokenizer, list_of_lists=args.do_line_break)

        print('caching...')
        input_df.to_json(sentences_fn, lines=True, orient='records')

    return list(input_df.to_dict(orient='records'))


spacy_model = None
def get_spacy_model():
    global spacy_model
    if spacy_model is None:
        import spacy
        print('loading spacy model...')
        spacy_model = spacy.load('en_core_web_lg')
        spacy_model.add_pipe('sentencizer')
    return spacy_model


def process_doc_sents(sents, tok, args):
    sents = list(flatten(map(lambda x: x.split('\n'), sents)))
    sents = list(map(lambda x: unidecode(x).strip(), sents))
    sents = list(filter(lambda x: x != '', sents))
    sents = list(map(lambda x: re.sub(r' +', ' ', x), sents))
    sent_lens = list(map(lambda x: len(tok.encode(x)), sents))
    cum_sent_lens = np.cumsum(sent_lens)
    start = np.where(cum_sent_lens > MAX_DOC_LENGTH)[0]
    if len(start) > 0:
        sents = sents[:start[0]]
    if len(sents) > args.max_num_sents:
        sents = sents[:args.max_num_sents]
    return sents


def sentencize_list(l, tok=None, verbose=True, process=True, args=None):
    to_disable = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "textcat"]
    spacy_model = get_spacy_model()
    pipe = spacy_model.pipe(l, disable=to_disable)
    if verbose:
        pipe = tqdm(pipe, total=len(l))
    doc_sentences = []
    for doc in pipe:
        sents = list(map(str, doc.sents))
        if process:
            sents = process_doc_sents(sents, tok, args)
        doc_sentences.append(sents)
    return doc_sentences


def merge_lowercase_sents(sent_list):
    """
    Iterate through a list of sentences, and merge any that are lowercase with the previous sentence.
    """
    merged_sents = []
    sent_list = list(filter(lambda x: x.strip() != '', sent_list))
    for sent in sent_list:
        if len(merged_sents) > 0 and sent[0].islower():
            merged_sents[-1] += ' ' + sent.strip()
        else:
            merged_sents.append(sent)
    return merged_sents


def sentencize_col(df, args, tokenizer, list_of_lists=False):
    """
    Sentencize a text column.
    If list_of_lists is True, then the "linebreak" operation is performed, which
    splits the text by linebreaks, and then sentencizes each line separately.

    This is useful for HTML documents, where linebreaks are often used to separate blurbs of text, like headers
    or buttons, which should be identified as separate sentences (and ideally cut out).

    We find that 1/8 of press releases have awkward midsentence linebreaks, so we have to additionally check to see
    if a high proportion of the sentence is lowercase, and if so, we merge it with the previous sentence.
    """

    print('sentencizing...')
    sents_to_get = df.loc[lambda df: df['sentences'].isnull()]
    if list_of_lists:
        doc_sentences = (
            sents_to_get[args.text_col]
                .str.split('\n')
                .progress_apply(merge_lowercase_sents)
                .progress_apply(lambda x: list(filter(is_acceptable_sentence, x)))
                .progress_apply(lambda x: list(unique_everseen(x)))
                .progress_apply(lambda x: list(flatten(sentencize_list(x, None, verbose=False, process=False, args=args))))
                .progress_apply(lambda x: process_doc_sents(x, tokenizer, args))
                .tolist()
        )
    else:
        doc_sentences = sentencize_list(sents_to_get[args.text_col], tokenizer, args=args)
    sentences_s = pd.Series(doc_sentences, index=sents_to_get[args.id_col]).to_frame('sentences')
    df = df.merge(sentences_s, left_on=args.id_col, right_index=True, how='left')
    df['sentences'] = df['sentences_x'].fillna(df['sentences_y'])
    df = df.drop(columns=['sentences_x', 'sentences_y'])
    return df


def load_model(model_name, model_class, config, distribute_model):
    return model_class.from_pretrained(model_name, config=config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--model_type', default='sentence', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--n-rows', default=None, type=int)
    parser.add_argument(
        '--sql-command',
        default='''
            SELECT common_crawl_url as article_url, article_text body 
            FROM article_data 
            WHERE is_press_release_article + is_archival_article = 1
            LIMIT 5
            ''',
        type=str
    )
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default='roberta-base', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--text-col', default='body', type=str)
    parser.add_argument('--id-col', default='suid', type=str)
    parser.add_argument('--do-line-break', action='store_true')
    parser.add_argument('--max-num-sents', default=MAX_NUM_SENTS, type=int)
    parser.add_argument('--distribute-model', action='store_true')
    args = parser.parse_args()

    # set naming
    tokenizer_name = get_tokenizer_name(args.model_name_or_path, args.tokenizer_name)
    model_class, dataset_class = get_model_and_dataset_class(args.model_type)

    # load model
    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path)
    model = load_model(
        args.model_name_or_path, model_class, config,
        args.distribute_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load in dataset
    data = load_data(args, tokenizer)
    dataset = dataset_class(
        tokenizer=tokenizer, do_score=True, label_mapper=label_mapper, max_length=MAX_SENT_LENGTH
    )
    device = get_device()
    model.eval()
    model = model.to(device)
    with open(args.outfile, 'w') as f:
        writer = jsonlines.Writer(f)
        for doc in tqdm(data, total=len(data)):
            print(len(doc['sentences']))
            if len(doc['sentences']) < 3:
                continue

            input_ids, attention_mask, _ = dataset.process_one_doc(doc)
            datum = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }

            # score
            scores = model.get_proba(**datum)
            preds = dataset.transform_logits_to_labels(scores, num_docs=len(input_ids))

            # process data
            output_datum = []
            for sent_idx, sent in enumerate(doc['sentences']):
                output_packet = {
                    'discourse_preds': preds[sent_idx],
                    'sentences': sent,
                    'doc_id': doc[args.id_col],
                    'sent_idx': sent_idx,
                }
                output_datum.append(output_packet)
            writer.write(output_datum)





# python predict.py --model_name_or_path alex2awesome/newsdiscourse-model --model_type sentence --dataset_name ../../../data/open-sourced-articles/all-articles.csv.gz --outfile ../data/all-news-discourse.jsonl --tokenizer_name roberta-base --text-col body --id-col article_url

# python predict.py --model_name_or_path alex2awesome/newsdiscourse-model --model_type sentence --dataset_name ../../../data/open-sourced-articles/all-articles-in-db.csv.gz --outfile ../data/all-news-discourse.jsonl --tokenizer_name roberta-base --text-col body --id-col  article_url --max-num-sents 50
# python predict.py --model_name_or_path alex2awesome/newsdiscourse-model --model_type sentence --dataset_name ../../../data/open-sourced-articles/all-articles-in-db.csv.gz --outfile ../data/all-news-discourse.jsonl --tokenizer_name roberta-base --text-col body --id-col  article_url --max-num-sents 50