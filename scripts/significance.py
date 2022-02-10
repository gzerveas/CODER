import os
import subprocess
import argparse

from scipy.stats import ttest_rel

MAX_MRR_RANK = 200


def save_sorted_results(results, file, until_rank=-1):
    with open(file, "w") as val_file:
        for qid in results.keys():
            query_data = list(results[qid].items())
            query_data.sort(key=lambda x: x[1], reverse=True)
            # sort the results per query based on the output
            for rank_i, (docid, score) in enumerate(query_data):
                #val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value])+"\n")
                val_file.write("%s Q0 %s %d %f neural\n" % (str(qid), str(docid), rank_i + 1, score))

                if until_rank > -1 and rank_i == until_rank + 1:
                    break


def parse_trec_eval(results_file):
    """Parse the output of trec_eval
    :param results_file:
    :return:
    """
    results_perq = {}
    results_avg = {}
    with open(results_file) as f:
        for line in f:
            values = line.strip('\n').strip('\r').split()
            if len(values) > 2:
                metric = values[0].strip()
                qid = values[1].strip()
                res = float(values[2].strip())
                if qid == 'all':
                    if metric not in results_avg:
                        results_avg[metric] = {}
                    results_avg[metric] = res
                else:
                    if metric not in results_perq:
                        results_perq[metric] = {}
                    results_perq[metric][qid] = res
    return  results_avg, results_perq


class EvaluationTool():
    def __init__(self):
        pass

    def evaluate(self, candidate, run_path_for_save, evalparam="-q", validaterun=False):
        pass


class EvaluationToolTrec(EvaluationTool):

    def __init__(self, trec_eval_path, qrel_path,
                 trec_measures_param="-m ndcg -m ndcg_cut -m recall -m recip_rank"):
        self.trec_eval_path = trec_eval_path
        self.qrel_path = qrel_path
        self.trec_measures_param = trec_measures_param

    def run_command(self, command):
        # p = subprocess.Popen(command.split(),
        #                     stdout=subprocess.PIPE,
        #                     stderr=subprocess.STDOUT)
        # return iter(p.stdout.readline, b'')
        stdoutdata = subprocess.getoutput(command)
        return stdoutdata.split('\n')

    def validate_correct_runfile(self, run_path):
        run_path_temp = run_path + '.temp'
        with open(run_path) as fr, open(run_path_temp, 'w') as fw:
            qid = 0
            docids = set([])
            for l in fr:
                vals = [x.strip() for x in l.strip().split()]
                if vals[0] != qid:
                    qid = vals[0]
                    docids = set([])
                if vals[2] in docids:
                    continue
                docids.add(vals[2])
                fw.write("%s %s %s %s %s %s\n" % (vals[0], vals[1], vals[2], len(docids), vals[4], vals[5]))
        os.remove(run_path)
        os.rename(run_path_temp, run_path)

    def evaluate(self, candidate, run_path_for_save=None, evalparam="-q", validaterun=False):

        if run_path_for_save is None:
            first_part, ext = os.path.splitext(candidate)
            run_path_for_save = first_part + '.sorted' + ext
        save_sorted_results(candidate, run_path_for_save)
        results_avg, results_perq = self.evaluate_from_file(candidate_path=run_path_for_save,
                                                            evalparam=evalparam, validaterun=validaterun)
        os.remove(run_path_for_save)

        return results_avg, results_perq

    def evaluate_from_file(self, candidate_path, evalparam="-q", validaterun=False):

        if validaterun:
            self.validate_correct_runfile(candidate_path)

        results_perq = {}
        results_avg = {}
        command = "%s %s %s %s %s" % (self.trec_eval_path, self.trec_measures_param, evalparam, self.qrel_path,
                                      candidate_path)
        print(command)
        for line in self.run_command(command):
            values = line.strip('\n').strip('\r').split()
            if len(values) > 2:
                metric = values[0].strip()
                qid = values[1].strip()
                res = float(values[2].strip())
                if qid == 'all':
                    if metric not in results_avg:
                        results_avg[metric] = {}
                    results_avg[metric] = res
                else:
                    if metric not in results_perq:
                        results_perq[metric] = {}
                    results_perq[metric][qid] = res

        return results_avg, results_perq


# class EvaluationToolMsmarco(EvaluationTool):
#
#     def __init__(self, qrel_path):
#         self.qrel_path = qrel_path
#         self.qids_to_relevant_docids = self.load_reference(self.qrel_path)
#
#     def load_reference_from_stream(self, f):
#         """Load Reference reference relevant documents
#         Args:f (stream): stream to load.
#         Returns:qids_to_relevant_docids (dict): dictionary mapping from query_id (int) to relevant documents (list of ints).
#         """
#         qids_to_relevant_docids = {}
#         for l in f:
#             vals = l.strip().split('\t')
#             if len(vals) != 4:
#                 vals = l.strip().split(' ')
#                 if len(vals) != 4:
#                     pdb.set_trace()
#                     raise IOError('\"%s\" is not valid format' % l)
#
#             qid = vals[0]
#             if qid in qids_to_relevant_docids:
#                 pass
#             else:
#                 qids_to_relevant_docids[qid] = []
#             _rel = int(vals[3])
#             if _rel > 0:
#                 qids_to_relevant_docids[qid].append(vals[2])
#
#         return qids_to_relevant_docids
#
#     def load_reference(self, path_to_reference):
#         """Load Reference reference relevant documents
#         Args:path_to_reference (str): path to a file to load.
#         Returns:qids_to_relevant_docids (dict): dictionary mapping from query_id (int) to relevant documents (list of ints).
#         """
#         with open(path_to_reference, 'r') as f:
#             qids_to_relevant_docids = self.load_reference_from_stream(f)
#         return qids_to_relevant_docids
#
#     def evaluate(self, candidate, run_path_for_save, evalparam=None, validaterun=False):
#
#         """Compute MRR metric
#         """
#         results_perq = {'recip_rank': {}}
#         results_avg = {'recip_rank': {}}
#         MRR = 0
#         ranking = []
#         for qid in candidate:
#             if qid in self.qids_to_relevant_docids:
#                 target_docid = self.qids_to_relevant_docids[qid]
#                 candidate_docidscore = list(candidate[qid].items())
#                 candidate_docidscore.sort(key=lambda x: x[1], reverse=True)
#                 candidate_docid = [x[0] for x in candidate_docidscore]
#                 ranking.append(0)
#                 results_perq['recip_rank'][qid] = 0
#                 # MRR
#                 for i in range(0, min(len(candidate_docid), MAX_MRR_RANK)):
#                     if candidate_docid[i] in target_docid:
#                         MRR += 1 / (i + 1)
#                         results_perq['recip_rank'][qid] = 1 / (i + 1)
#                         ranking.pop()
#                         ranking.append(i + 1)
#                         break
#
#         if len(ranking) == 0:
#             raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
#
#         results_avg['recip_rank'] = MRR / len(ranking)
#
#         return results_avg, results_perq


# def store_sorted(results_perq):
#     evalres[_folder][_filename] = {}
#     qryids = None
#     for _id in runeval[_folder][_filename]:
#         # print (_filename, _id)
#         evalres[_folder][_filename][_id] = {}
#         for _metric in runeval[_folder][_filename][_id]['metrics_perq']:
#             if qryids == None:
#                 qryids = list(runeval[_folder][_filename][_id]['metrics_perq'][_metric].keys())
#                 qryids.sort()
#                 print(len(qryids))
#
#             evalres[_folder][_filename][_id][_metric] = [0.0, [],
#                                                          []]  # [eval result, per query results, sig improvement ids]
#             evalres[_folder][_filename][_id][_metric][0] = runeval[_folder][_filename][_id]['metrics_avg'][_metric]
#             evalres[_folder][_filename][_id][_metric][1] = [runeval[_folder][_filename][_id]['metrics_perq'][_metric][x]
#                                                             for x in qryids]
#
#
# evalres = {}
# for _folder in results_folders:
#     print(_folder)
#     evalres[_folder] = {}
#     for _filename in runeval[_folder]:
#         print(_filename)
#         evalres[_folder][_filename] = {}
#         qryids = None
#         for _id in runeval[_folder][_filename]:
#             # print (_filename, _id)
#             evalres[_folder][_filename][_id] = {}
#             for _metric in runeval[_folder][_filename][_id]['metrics_perq']:
#                 if qryids == None:
#                     qryids = list(runeval[_folder][_filename][_id]['metrics_perq'][_metric].keys())
#                     qryids.sort()
#                     print(len(qryids))
#
#                 evalres[_folder][_filename][_id][_metric] = [0.0, [], []]  # [eval result, per query results, sig improvement ids]
#                 evalres[_folder][_filename][_id][_metric][0] = runeval[_folder][_filename][_id]['metrics_avg'][_metric]
#                 evalres[_folder][_filename][_id][_metric][1] = [runeval[_folder][_filename][_id]['metrics_perq'][_metric][x] for x in qryids]
#
# print('done!')
#
# ## calculating Significance test
# for _folder in results_folders:
#     print(_folder)
#     for _filename in evalres[_folder]:
#         print(_filename)
#         for _id in evalres[_folder][_filename]:
#             # print (_filename, _id)
#             for _id_base in evalres[_folder][_filename]:
#                 if (_id == _id_base):
#                     continue
#                 for _metric in _measures_sorted:
#                     if (evalres[_folder][_filename][_id][_metric][0] <= evalres[_folder][_filename][_id_base][_metric][0]):
#                         continue
#                     _lst = evalres[_folder][_filename][_id][_metric][1]
#                     _lst_base = evalres[_folder][_filename][_id_base][_metric][1]
#                     statistic, pvalue = ttest_rel(_lst, _lst_base)
#                     # print (_id, _id_base, pvalue)
#                     if pvalue < 0.05:
#                         evalres[_folder][_filename][_id][_metric][2].append((_id_base, pvalue))
#                     # if _metric=='recip_rank':
#                     #    print (_id, _id_base, pvalue)
# print('done!')



if __name__ == "__main__":

    METRICS = ['ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100', 'recall_1000']

    parser = argparse.ArgumentParser("Run significance tests")

    parser.add_argument("--trec_eval_binary", type=str,
                        help="File path of trec_eval binary")
    parser.add_argument("--base", type=str,
                        help="Path to rankings by model/system with respect to which we want to test whether improvement is significant")
    parser.add_argument("--contender", type=str,
                        help="Path to rankings by model/system which should improve on the base model")
    parser.add_argument("--qrels_path", type=str,
                        help="""Path to qrels (ground truth relevance judgements) as needed by trec_eval""")
    args = parser.parse_args()

    print("Base: {}".format(args.base))
    print("Contender: {}".format(args.contender))
    if args.trec_eval_binary is not None:
        evaluator = EvaluationToolTrec(args.trec_eval_binary, args.qrels_path)
        print("Running trec_eval on {} ...".format(args.base))
        base_results_avg, base_results_perq = evaluator.evaluate(args.base)
        print("Running trec_eval on {} ...".format(args.contender))
        contend_results_avg, contend_results_perq = evaluator.evaluate(args.contender)
    else:
        print("Parsing evaluation results in {} ...".format(args.base))
        base_results_avg, base_results_perq = parse_trec_eval(args.base)
        print("Parsing evaluation results in {} ...".format(args.contender))
        contend_results_avg, contend_results_perq = parse_trec_eval(args.contender)

    print()
    for metric in METRICS: #contend_results_perq:
        print("\nMETRIC: {}".format(metric))
        num_base_values = len(base_results_perq[metric])
        num_contender_values = len(contend_results_perq[metric])
        if num_contender_values != num_base_values:
            print("WARNING: {} query IDs in contender, but {} in base!".format(num_contender_values, num_base_values))
        qids = sorted(contend_results_perq[metric].keys())
        base_values = [base_results_perq[metric][qid] for qid in qids]
        contend_values = [contend_results_perq[metric][qid] for qid in qids]
        base_avg = sum(base_values)/num_base_values
        print("Base avg: {:.3f} (avg in file: {:.3f})".format(base_avg, base_results_avg[metric]))
        contender_avg = sum(contend_values)/num_contender_values
        print("Contender avg: {:.3f} (avg in file: {:.3f})".format(contender_avg, contend_results_avg[metric]))

        statistic, pvalue = ttest_rel(contend_values, base_values)
        sig_string = "SIGNIFICANT" if pvalue < 0.05 else "NOT significant"
        print("P value: {} - {}".format(pvalue, sig_string))


