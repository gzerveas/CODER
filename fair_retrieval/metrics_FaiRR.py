import argparse
import numpy as np
import pickle
import pdb
import itertools
import copy

class FaiRRMetric:
    
    def __init__(self, collection_neutrality_path, background_doc_set):
        self.documents_neutrality = {}
        for l in open(collection_neutrality_path):
            vals = l.strip().split('\t')
            self.documents_neutrality[int(vals[0])] = float(vals[1])
        self.background_doc_set = background_doc_set
        
    # the normalization term IFaiRR is calculated using the documents of the to retrieval_results
    # retrieval_results : a dictionary with queries and the ordered lists of documents
    # background_doc_set : a dictionary with queries and the set of background documents
    def calc_FaiRR_retrievalresults(self, retrievalresults, thresholds=[5,10,20,50]):
        
        _position_biases = [1/(np.log2(_rank+1)) for _rank in range(1, np.max(thresholds)+1)]

        ## get neutrality of documents
        _retres_neut = {}
        for _qryid in retrievalresults:
            _retres_neut[_qryid] = []
            for _docid in retrievalresults[_qryid][:np.max(thresholds)]:
                if _docid in self.documents_neutrality:
                    _neutscore = self.documents_neutrality[_docid]
                else:
                    _neutscore = 1.0
                    print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
                _retres_neut[_qryid].append(_neutscore)
        
        _bachgroundset_neut = {}
        for _qryid in self.background_doc_set:
            _bachgroundset_neut[_qryid] = []
            for _docid in self.background_doc_set[_qryid]:
                if _docid in self.documents_neutrality:
                    _neutscore = self.documents_neutrality[_docid]
                else:
                    _neutscore = 1.0
                    print("WARNING: Document neutrality score of ID %d is not found (set to 1)" % _doc_id)
                _bachgroundset_neut[_qryid].append(_neutscore)
        
        ## calculate FaiRR
        FaiRR = {}
        FaiRR_perq = {}
        for _threshold in thresholds:
            FaiRR_perq[_threshold] = {}
            for _qryid in _retres_neut:
                _th = np.min([len(_retres_neut[_qryid]), _threshold])
                FaiRR_perq[_threshold][_qryid] = np.sum(np.multiply(_retres_neut[_qryid][:_th], _position_biases[:_th]))
            FaiRR[_threshold] = np.mean(list(FaiRR_perq[_threshold].values()))

        ## calculate Ideal FaiRR
        IFaiRR_perq = {}
        for _qryid in _bachgroundset_neut:
            _bachgroundset_neut[_qryid].sort(reverse=True)
        for _threshold in thresholds:
            IFaiRR_perq[_threshold] = {}
            for _qryid in _bachgroundset_neut:
                _th = np.min([len(_bachgroundset_neut[_qryid]), _threshold])
                IFaiRR_perq[_threshold][_qryid] = np.sum(np.multiply(_bachgroundset_neut[_qryid][:_th], _position_biases[:_th]))
            
        ## calculate Normalized FaiRR
        #pdb.set_trace()
        NFaiRR = {}
        NFaiRR_perq = {}
        for _threshold in thresholds:
            NFaiRR_perq[_threshold] = []
            for _qryid in FaiRR_perq[_threshold]:
                if _qryid not in IFaiRR_perq[_threshold]:
                    print("ERROR: query id %d does not exist in background document set. Error ignored" % _qryid)
                    continue
                NFaiRR_perq[_threshold].append(FaiRR_perq[_threshold][_qryid] / IFaiRR_perq[_threshold][_qryid])
            NFaiRR[_threshold] = np.mean(NFaiRR_perq[_threshold])
        
        return FaiRR, NFaiRR

class FaiRRMetricHelper:

    def read_retrievalresults_from_runfile(self, trec_run_path, cut_off=200):
        retrievalresults = {}
        
        print ("Reading %s" % trec_run_path)

        with open(trec_run_path) as fr:
            qryid_cur = 0
            for i, line in enumerate(fr):
                vals = line.strip().split(' ')
                if len(vals) != 6:
                    vals = line.strip().split('\t')
                
                if len(vals) == 6:
                    _qryid = int(vals[0].strip())
                    _docid = int(vals[2].strip())
                    
                    if _qryid != qryid_cur:
                        retrievalresults[_qryid] = []
                        qryid_cur = _qryid

                    if len(retrievalresults[_qryid]) < cut_off:
                        retrievalresults[_qryid].append(_docid)
                else:
                    pass

        print ('%d lines read. Number of queries: %d' % (len(list(itertools.chain.from_iterable(retrievalresults.values()))), 
                                                         len(retrievalresults.keys())))
        
        return retrievalresults
    
    def read_documentset_from_retrievalresults(self, trec_run_path):
        _retrivalresults_background = self.read_retrievalresults_from_runfile(trec_run_path)
        background_doc_set = {}
        for _qryid in _retrivalresults_background:
            background_doc_set[_qryid] = set(_retrivalresults_background[_qryid])
        return background_doc_set


if __name__ == "__main__":
    #
    # config
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--collection-neutrality-path', action='store', dest='collection_neutrality_path',
                        default="/share/cp/datasets/ir/msmarco/passage/fair-retrieval-results/collection_with_neutrality.tsv",
                        help='path to the file containing neutrality values of documents in tsv format (docid [tab] score)')
    parser.add_argument('--runfile', action='store', dest='runfile',
                        help='path to the run file in TREC format', required=True)
    parser.add_argument('--backgroundrunfile', action='store', default="resources/msmarco_fair.background_run.txt",
                        help='path to the run file for the set of background documents in TREC format', required=True)
    args = parser.parse_args()
    
    _metrichelper = FaiRRMetricHelper()
    _retrivalresults = _metrichelper.read_retrievalresults_from_runfile(args.runfile)
    _background_doc_set = _metrichelper.read_documentset_from_retrievalresults(args.backgroundrunfile)
    
    _metric = FaiRRMetric(args.collection_neutrality_path, _background_doc_set)
    _metric_res = _metric.calc_FaiRR_retrievalresults(_retrivalresults)
    
    print (_metric_res)
    