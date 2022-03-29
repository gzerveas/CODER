import time
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from modeling import get_aggregation_function
from dataset import load_original_sequences, EmbeddedSequences
from precompute import MSMARCO_DocDataset
import utils
from utils import rank_docs, get_relevances, calculate_metrics

logger = logging.getLogger(__name__)


def inspect(args, model, dataloader):
    """
    Interactively examine an existing ranking of candidates by the specified model, alongside with the respective
    original queries and documents, reconstructed tokenizations, embeddings, ground truth relevant documents,
    and a (raw .tsv) reference ranked candidates file, which may or may not correspond to the same ranking
    as `eval_candidates_path`
    """

    # to have agreement between loaded and computed embeddings, sequence lengths must correspond to the max. length used for precomputed embeddings
    QUERY_LENGTH = args.max_query_length
    DOC_LENGTH = args.max_doc_length
    DISPLAY_DOCS = 10  # how many candidate documents (from the top and bottom) to display and use for integrity calculation
    ERROR_TOLERANCE = 1e-4  # per element tolerable absolute difference in embeddings
    np.set_printoptions(linewidth=200)
    torch.set_printoptions(linewidth=200)
    pd.options.display.width = 400  # will automatically detect console width
    pd.set_option('display.max_colwidth', 0)
    autorun = False  # if True, will iterate over entire eval set without stopping

    # Modify existing console handler
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    qrels = dataloader.dataset.qrels  # dict{qID: dict{pID: relevance}}
    labels_exist = qrels is not None

    # num_docs is the (potentially variable) number of candidates per query
    relevances = []  # (total_num_queries) list of (num_docs) lists with non-zeros at the indices corresponding to actually relevant passages
    num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
    df_chunks = []  # (total_num_queries) list of dataframes, each with index a single qID and corresponding (num_docs) columns PID, rank, score
    query_time = 0  # average time for the model to score candidates for a single query
    total_loss = 0  # total loss over dataset
    error_queries = []  # query IDs corresponding to tokenization or embedding errors
    error_docs = set([])  # document IDs corresponding to tokenization or embedding errors
    num_inspected_docs = 0  # total number of inspected documents
    total_num_score_errors = 0  # total number of errors in scores
    total_num_rank_violations = 0  # total number of ranks affected by differences in scores
    total_queries = 0  # total number of inspected queries

    logger.info("Loading raw queries ...")
    query_dict = load_original_sequences(args.raw_queries_path)
    if args.query_emb_memmap_dir:
        logger.info("Loading query embeddings from '{}' ...".format(args.query_emb_memmap_dir))
        query_embeddings = EmbeddedSequences(embedding_memmap_dir=args.query_emb_memmap_dir, seq_type='query')

    logger.info("Loading raw documents ...")
    doc_dict = load_original_sequences(args.raw_collection_path)

    if args.collection_memmap_dir:
        logger.info("Loading tokenized documents collection from '{}' ...".format(args.collection_memmap_dir))
        doc_tokens_collection = MSMARCO_DocDataset(args.collection_memmap_dir, DOC_LENGTH, dataloader.dataset.tokenizer)

    aggregation_func = get_aggregation_function(args.query_aggregation)

    with torch.no_grad():
        for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            start_time = time.perf_counter()
            out = model(**batch_data)
            query_time += time.perf_counter() - start_time
            rel_scores = out['rel_scores'].detach().cpu().numpy()  # (batch_size, num_docs) relevance scores in [0, 1]
            if 'loss' in out:
                total_loss += out['loss'].sum().item()
            assert len(qids) == len(docids) == len(rel_scores)

            # Calculate and extract encoder final query representations
            encoder_out = model.encoder(batch_data['query_token_ids'].to(torch.int64),
                                        attention_mask=batch_data['query_mask'])
            enc_hidden_states = encoder_out['last_hidden_state']  # (batch_size, max_query_len, query_dim)

            # Rank documents based on their scores
            num_docs_per_query = [len(cands) for cands in docids]
            num_lengths = set(num_docs_per_query)
            no_padding = (len(num_lengths) == 1)  # whether all queries in this batch had the same number of candidates

            if no_padding:  # (only) 10% speedup compared to other case
                docids_array = np.array(docids, dtype=np.int32)  # (batch_size, num_docs) array of docIDs per query
                # First shuffle along doc dimension, because relevant document(s) are placed at the beginning and would benefit
                # in case of score ties! (can happen e.g. with saturating score functions)
                inds = np.random.permutation(rel_scores.shape[1])
                np.take(rel_scores, inds, axis=1, out=rel_scores)
                np.take(docids_array, inds, axis=1, out=docids_array)

                # Sort by descending relevance
                inds = np.fliplr(np.argsort(rel_scores, axis=1))  # (batch_size, num_docs) inds to sort rel_scores
                # (batch_size, num_docs) docIDs per query, in order of descending relevance score
                ranksorted_docs = np.take_along_axis(docids_array, inds, axis=1)
                sorted_scores = np.take_along_axis(rel_scores, inds, axis=1)
            else:
                # (batch_size) iterables of docIDs and scores per query, in order of descending relevance score
                ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

            # extend by batch_size elements
            batch_dfs = [pd.DataFrame(data={"PID": ranksorted_docs[i],
                                            "rank": list(range(1, len(docids[i]) + 1)),
                                            "score": sorted_scores[i]},
                                      index=[qids[i]] * len(docids[i])) for i in range(len(qids))]
            df_chunks.extend(batch_dfs)

            if labels_exist:
                relevances.extend(get_relevances(qrels[qids[i]], ranksorted_docs[i]) for i in range(len(qids)))
                num_relevant.extend(len([docid for docid in qrels[qid] if qrels[qid][docid] > 0]) for qid in qids)

            # Display input and output information for retrieval on each query
            for i in range(len(qids)):  # iterate over batch_size
                total_queries += 1
                # Check integrity of queries
                query_ok = True
                logger.info("Query ID: {}".format(qids[i]))
                logger.info("Original Text: {}".format(query_dict[qids[i]]))

                logger.debug("Input query token IDs (as in '{}'):\n{}".format(args.tokenized_path,
                                                                              batch_data['query_token_ids'][i, :]))
                logger.debug("Rec. input query tokens:\n{}".format(
                    dataloader.dataset.tokenizer.convert_ids_to_tokens(batch_data['query_token_ids'][i, :])))
                start_time = time.perf_counter()
                tokenized_query = dataloader.dataset.tokenizer([query_dict[qids[i]]], max_length=QUERY_LENGTH, padding=True, return_tensors="pt")
                tokenization_time = time.perf_counter() - start_time
                logger.debug("Time to tokenize: {} s".format(tokenization_time))
                tokenized_query = {k: v.to(args.device) for k, v in tokenized_query.items()}
                qtokenized_ids = tokenized_query["input_ids"][0]
                logger.debug("Tokenized query IDs: {}".format(qtokenized_ids))
                qtokenized_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(qtokenized_ids)
                logger.debug("Query tokens: {}".format(qtokenized_tokens))

                if not (torch.equal(qtokenized_ids, batch_data['query_token_ids'][i, :len(qtokenized_ids)])):
                    logger.warning("Query tokenization error!")
                    query_ok = False

                query_emb_model = aggregation_func(enc_hidden_states)[i, :]  # .detach().cpu().numpy()
                logger.debug("Query embedding (inside model):\n{}".format(query_emb_model))
                encoder_out2 = model.encoder(**tokenized_query)  # using currently tokenized query
                query_emb_model2 = aggregation_func(encoder_out2[0])  # [0] selects 'encoder_hidden_states'
                abs_diffs = torch.abs(query_emb_model - query_emb_model2)
                max_abs_diff = torch.max(abs_diffs).detach().cpu().numpy()
                # L2_diff = torch.linalg.norm(emb_diff).detach().cpu().numpy()
                mean_abs_diff = torch.mean(abs_diffs).detach().cpu().numpy()
                if max_abs_diff > ERROR_TOLERANCE:
                    logger.debug("Query embedding (directly from encoder):\n{}".format(query_emb_model2))
                    logger.warning("Query embedding as calculated from pre-tokenized query inside model differs from "
                                   "the embedding as calculated directly by encoder!")
                    logger.warning("Max. abs. difference: {} \t Mean abs. difference: {}".format(max_abs_diff,
                                                                                                 mean_abs_diff))
                    error_inds = abs_diffs > ERROR_TOLERANCE  # (num_docs, emb_dim)
                    logger.warning("{} / {} elements have an absolute "
                                   "difference larger than {}".format(torch.sum(error_inds),
                                                                      torch.numel(abs_diffs), ERROR_TOLERANCE))
                    query_ok = False

                if args.query_emb_memmap_dir:
                    loaded_query_emb = torch.Tensor(query_embeddings[[qids[i]]]).to(args.device)
                    logger.debug("Loaded query embedding (from '{}'):\n{}".format(args.query_emb_memmap_dir, loaded_query_emb))
                    abs_diffs = torch.abs(query_emb_model2 - loaded_query_emb)
                    max_abs_diff = torch.max(abs_diffs).detach().cpu().numpy()
                    #L2_diff = torch.linalg.norm(emb_diff).detach().cpu().numpy()
                    mean_abs_diff = torch.mean(abs_diffs).detach().cpu().numpy()
                    if max_abs_diff > ERROR_TOLERANCE:
                        logger.warning("Query embedding as CALCULATED directly by encoder differs from "
                                       "the embedding as LOADED from '{}'!".format(args.query_emb_memmap_dir))
                        logger.warning("Max. abs. difference: {} \t Mean abs. difference: {}".format(max_abs_diff,
                                                                                                     mean_abs_diff))
                        error_inds = abs_diffs > ERROR_TOLERANCE  # (num_docs, emb_dim)
                        logger.warning("{} / {} elements have an absolute "
                                       "difference larger than {}".format(torch.sum(error_inds),
                                                                          torch.numel(abs_diffs), ERROR_TOLERANCE))
                        query_ok = False

                if not query_ok:
                    error_queries.append(qids[i])

                # Check integrity of documents and scores
                # Checks DISPLAY_DOCS top and bottom candidate docs, and ground-truth relevant docs, if they exist
                docs_df = batch_dfs[i]  # df with PIDs, ranks, scores for a single qID
                ranksorted_docids = list(ranksorted_docs[i])
                docs_text = [doc_dict[did] for did in ranksorted_docids]  # list of candidate doc texts (for 1 query)
                docs_df['text'] = docs_text
                docs_df['text'] = docs_df['text'].str.wrap(200)  # to set max line width for text (wrap)
                docs_to_check_text = []  # list of candidate doc texts to be checked
                docs_to_check_ids = []  # list of candidate doc IDs to be checked

                if labels_exist:
                    gt_docids = [docid for docid in qrels[qids[i]] if qrels[qids[i]][docid] > 0]
                    gt_doc_text = []
                    logger.info("Ground truth relevant document(s): ")
                    for d, did in enumerate(gt_docids):
                        print("{}: DocID:{:8}: {}\n".format(d, did, doc_dict[did]))
                        gt_doc_text.append(doc_dict[did])
                    docs_to_check_text = gt_doc_text
                    docs_to_check_ids = gt_docids

                logger.info("Top-Scored candidates:\n{}\n".format(docs_df.iloc[:DISPLAY_DOCS]))
                logger.info("Bottom-Scored candidates:\n{}\n".format(docs_df.iloc[-DISPLAY_DOCS:]))

                docs_to_check_text += docs_text[:DISPLAY_DOCS] + docs_text[-DISPLAY_DOCS:]
                docs_to_check_ids += ranksorted_docids[:DISPLAY_DOCS] + ranksorted_docids[-DISPLAY_DOCS:]
                num_inspected_docs += len(docs_to_check_ids)
                start_time = time.perf_counter()
                tokenized_docs = dataloader.dataset.tokenizer(docs_to_check_text, max_length=DOC_LENGTH, padding=True, return_tensors="pt")
                tokenization_time = time.perf_counter() - start_time
                logger.debug("Time to tokenize {} documents: {} s".format(len(docs_to_check_text), tokenization_time))

                for j in range(tokenized_docs["input_ids"].shape[0]):
                    logger.debug("DocID {}:\n".format(docs_to_check_ids[j]))
                    doc_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(tokenized_docs["input_ids"][j])
                    logger.debug("Tokenized: {}\n".format(doc_tokens))
                    if args.collection_memmap_dir:
                        loaded_token_ids = torch.tensor(doc_tokens_collection[docs_to_check_ids[j]]['input_ids'])
                        loaded_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(loaded_token_ids)
                        logger.debug("Loaded: {}\n\n".format(loaded_tokens))
                        if not (torch.equal(tokenized_docs["input_ids"][j][:len(loaded_token_ids)], loaded_token_ids)):
                            logger.warning("Doc tokenization error!")
                            error_docs.add(docs_to_check_ids[j])

                tokenized_docs = {k: v.to(args.device) for k, v in tokenized_docs.items()}
                model_out = model.encoder(**tokenized_docs)
                doc_emb_model = aggregation_func(model_out[0])  # (num_check_docs, d_emb) . [0] selects 'encoder_hidden_states'
                loaded_doc_emb = torch.Tensor(dataloader.dataset.emb_collection[docs_to_check_ids]).to(args.device)  # (num_check_docs, d_emb)
                abs_diffs = torch.abs(doc_emb_model - loaded_doc_emb)
                max_abs_diff = torch.max(abs_diffs)

                if max_abs_diff > ERROR_TOLERANCE:
                    logger.warning("Doc embeddings as COMPUTED directly by model differ from "
                                   "the embeddings as LOADED from '{}'!".format(args.embedding_memmap_dir))
                    logger.warning("Max. abs. difference: {} \t Mean abs. difference: {}".format(max_abs_diff, torch.mean(abs_diffs)))
                    error_inds = abs_diffs > ERROR_TOLERANCE  # (num_docs, emb_dim)
                    logger.warning("{} / {} elements have an absolute "
                                   "difference larger than {}".format(torch.sum(error_inds), torch.numel(doc_emb_model), ERROR_TOLERANCE))
                    error_inds = torch.any(error_inds, dim=1)  # (num_docs,) boolean array showing which docs differed
                    error_docs.update(docs_to_check_ids[k] for k in range(len(error_inds)) if error_inds[k])
                    logger.debug("Loaded doc embeddings (from '{}'):\n{}".format(args.embedding_memmap_dir, loaded_doc_emb))
                    logger.debug("Computed doc embeddings:\n{}".format(doc_emb_model))

                # Compare scores between MDST model output and computed encoder scores (emb. dot product)
                docs_to_check_inds = list(np.nonzero(relevances[i])[0]) if labels_exist else []  # insert indices of gt documents
                docs_to_check_inds += list(range(DISPLAY_DOCS)) + list(range(-DISPLAY_DOCS, 0, 1))
                model_scores = sorted_scores[i][docs_to_check_inds]  # scores as calculated in MDST model
                encoder_scores = torch.matmul(query_emb_model2, doc_emb_model.T).squeeze().detach().cpu()  # encoder scores (emb. dot product)
                direct_ranking = torch.argsort(encoder_scores[1:], descending=True).numpy()  # [1:] to exclude g.t.
                num_rank_violations = np.sum(np.abs(direct_ranking - np.arange(len(direct_ranking))))
                total_num_rank_violations += num_rank_violations
                encoder_scores = encoder_scores.numpy()
                # encoder_scores = torch.matmul(query_emb_model2, loaded_doc_emb.T).detach().cpu().numpy()  # encoder scores (emb. dot product)  # TODO: just for DEBUG, to confirm that scoring module does simple dot product
                abs_diffs = np.abs(model_scores - encoder_scores)
                max_abs_diff = np.max(abs_diffs)

                if max_abs_diff > ERROR_TOLERANCE:
                    logger.warning("Scores as computed directly by encoder model (dot product) differ from "
                                   "the scores in the output of MDST model.")
                    logger.warning("Max. abs. difference: {} \t Mean abs. difference: {}".format(max_abs_diff, np.mean(abs_diffs)))
                    error_inds = abs_diffs > ERROR_TOLERANCE
                    num_score_errors = np.sum(error_inds)
                    total_num_score_errors += num_score_errors

                    logger.warning("{} / {} scores have an absolute difference larger than {}".format(num_score_errors, len(model_scores), ERROR_TOLERANCE))
                    logger.warning("Number of ranks violations induced by difference: {}".format(num_rank_violations))
                    logger.info("Scores in the output of MDST model:\n{}\n".format(model_scores))
                    logger.info("Scores computed by encoder model (query-doc emb. dot product):\n{}\n".format(encoder_scores))

                if not autorun:
                    print()
                    inp = input("Press any key to inspect next query, 'c' to run through entire eval set, or 'q' to exit: ")
                    if inp == 'q':
                        break
                    elif inp == 'c':
                        autorun = True
                        continue
            if inp == 'q':
                break

    if len(error_queries):
        logger.error("Errors found in {} "
                     "out of {} ({:.3f}%) queries:\n{}".format(len(error_queries), total_queries,
                                                               100*len(error_queries)/total_queries,
                                                               error_queries))
        for q in error_queries:
            logger.debug("QID:{:7}: {}".format(q, query_dict[q]))
    else:
        logger.info("All {} queries ok!".format(len(dataloader.dataset.qids)))
    print()

    if len(error_docs):
        logger.error("Errors found in {} "
                     "out of {} ({:.3f}%) inspected documents:\n{}".format(len(error_docs), num_inspected_docs,
                                                                           100*len(error_docs)/num_inspected_docs,
                                                                           error_docs))
        for d in error_docs:
            logger.debug("docID:{:8}: {}".format(d, doc_dict[d]))
    else:
        logger.info("All {} documents ok!".format(num_inspected_docs))
    print()

    if total_num_score_errors:
        logger.error("Errors found in {} score calculations".format(total_num_score_errors))
        logger.error("{} ranks would be affected in total "
                     "({:.3f} per query)".format(total_num_rank_violations, total_num_rank_violations/total_queries))
    else:
        logger.info("All scores are consistent!")

    if labels_exist:
        try:
            eval_metrics = calculate_metrics(relevances, num_relevant, args.metrics_k)  # aggr. metrics for the entire dataset
        except:
            logger.error('Metrics calculation failed!')
            eval_metrics = OrderedDict()
        eval_metrics['loss'] = total_loss / len(dataloader.dataset)  # average over samples
    else:
        eval_metrics = OrderedDict()
    eval_metrics['query_time'] = query_time / len(dataloader.dataset)  # average over samples
    ranked_df = pd.concat(df_chunks, copy=False)  # index: qID (shared by multiple rows), columns: PID, rank, score

    if labels_exist and (args.debug or args.task != 'train'):
        try:
            rs = (np.nonzero(r)[0] for r in relevances)
            ranks = [1 + int(r[0]) if r.size else 1e10 for r in rs]  # for each query, what was the rank of the rel. doc
            freqs, bin_edges = np.histogram(ranks, bins=[1, 5, 10, 20, 30] + list(range(50, 1050, 50)))
            bin_labels = ["[{}, {})".format(bin_edges[i], bin_edges[i + 1])
                          for i in range(len(bin_edges) - 1)] + ["[{}, inf)".format(bin_edges[-1])]
            logger.info('\nHistogram of ranks for the ground truth documents:\n')
            utils.ascii_bar_plot(bin_labels, freqs, width=50, logger=logger)
        except:
            logger.error('Not possible!')

    return eval_metrics, ranked_df
