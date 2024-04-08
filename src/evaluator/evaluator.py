from typing import Dict, List, Union, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import logging
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from .eval_utils import parse_text_with_citations

logger = logging.getLogger(__name__)

class Evaluator(nn.Module):
    def __init__(
        self, 
        claimsplit_model_path: str = None,
        nli_model_path: str = None,
        device_map: Optional[str] = 'cuda',
        half_precision: bool = True,
    ):
        super().__init__()
        self.nli_model_path = nli_model_path
        self.claimsplit_model_path = claimsplit_model_path

        # load claim-split model
        if claimsplit_model_path is not None:
            config = AutoConfig.from_pretrained(claimsplit_model_path, trust_remote_code=True)
            torch_dtype = torch.bfloat16 if half_precision and config.torch_dtype==torch.float32 else config.torch_dtype
            self.claimsplit_model = AutoModelForSeq2SeqLM.from_pretrained(
                claimsplit_model_path, config=config, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
            )
            self.claimsplit_model.eval()
            self.claimsplit_tokenizer = AutoTokenizer.from_pretrained(claimsplit_model_path, trust_remote_code=True)

        else:
            self.claimsplit_model = None
            self.claimsplit_tokenizer = None
            

        # load nli model
        if nli_model_path is not None:
            config = AutoConfig.from_pretrained(nli_model_path, trust_remote_code=True)
            torch_dtype = torch.bfloat16 if half_precision and config.torch_dtype==torch.float32 else config.torch_dtype

            if config.model_type in ['t5', 'mt5']: # seq2seq model
                if "mt5-large-finetuned-mnli-xtreme-xnli" in nli_model_path:
                    config.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
                    config.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
                elif "t5_xxl_true_nli_mixture" in nli_model_path:
                    # this model does binary classification, and we set contradict label to -1 which will be never predicted
                    config.id2label = {1: 'entailment', 0: 'neutral', -1: 'contradiction'}
                    config.label2id = {'entailment': 1, 'neutral': 0, 'contradiction': -1}
                self.nli_model = AutoModelForSeq2SeqLM.from_pretrained(
                    nli_model_path, config=config, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
                )
            else:   # bert-like model
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    nli_model_path, config=config, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
                 )
            self.nli_model.eval()
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path, trust_remote_code=True)  

            for label in config.id2label.values():
                if 'entailment' in label.lower():
                    self.entail_label = label
                elif 'neutral' in label.lower():
                    self.neutral_label = label
                elif 'contradiction' in label.lower():
                    self.contradict_label = label
        else:
            self.nli_model = None
            self.nli_tokenizer = None
            
    @property
    def id2label(self):
        return self.nli_model.config.id2label
    @property
    def label2id(self):
        return self.nli_model.config.label2id
    @property
    def num_labels(self):
        return self.nli_model.config.num_labels 

    
    @torch.inference_mode()
    def _predict_nli(
        self,
        hypothesis: Union[str, List[str]], 
        premise: Union[str, List[str]],
        max_length: int = 512,
    ) -> List[int]:
        
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        if isinstance(premise, str):
            premise = [premise]
        assert len(hypothesis) == len(premise), f"the number of hypotheses is {len(hypothesis)} but the number of premises is {len(premise)}"

        if 'bert' in self.nli_model.config.model_type.lower(): # use hiddent state of cls_token to classify
            inputs = self.nli_tokenizer(
                premise, hypothesis, padding='longest', truncation=True, max_length=max_length, return_tensors='pt'
            ).to(self.nli_model.device)

            logits = self.nli_model(**inputs).logits 
            predictions = torch.argmax(logits, dim=-1).tolist()

         # convert the nli task into seq2seq generation
        elif "mt5-large-finetuned-mnli-xtreme-xnli" in self.nli_model_path:
            # reference: https://huggingface.co/alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli/blob/main/README.md
            prompt_template = "xnli: premise: {} hypothesis: {}"
            input_text = [prompt_template.format(p, h) for p, h in zip(premise, hypothesis)]
            inputs = self.nli_tokenizer(
                input_text, padding='longest', truncation=True, max_length=max_length, return_tensors='pt'
            ).to(self.nli_model.device)

            out = self.nli_model.generate(
                **inputs, output_scores=True, max_new_tokens=3, return_dict_in_generate=True, num_beams=1
            ) 
            # sanity check that our sequences are expected length (1 + start token + end token = 3)
            for i, seq in enumerate(out.sequences):
                assert len(seq) == 3, f"generated sequence {i} not of expected length 3,  Actual length: {len(seq)}"
            predictions = [int(pred) for pred in self.nli_tokenizer.batch_decode(out.sequences, skip_special_tokens=True)]
        
        elif "t5_xxl_true_nli_mixture" in self.nli_model_path:
            # reference: https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py 
            prompt_template = "premise: {} hypothesis: {}"
            input_text = [prompt_template.format(p, h) for p, h in zip(premise, hypothesis)]
            inputs = self.nli_tokenizer(
                input_text, padding='longest', truncation=True, max_length=max_length, return_tensors='pt'
            ).to(self.nli_model.device)
            out = self.nli_model.generate(**inputs, max_new_tokens=3)
            predictions = self.nli_tokenizer.batch_decode(out, skip_special_tokens=True)
            predictions = [1 if pred=="1" else 0 for pred in predictions]
        
        return predictions

        
    @torch.inference_mode()
    def _split_claims(
        self,
        sentences: Union[str, List[str]],
        max_source_length=64,
        max_target_length=128,
        batch_size=256,
        task_prefix='claim-split: ',
        verbose=False
    ) -> Union[List[str], List[List[str]]]:
        """
        Split the source sentence into a list of claims using the claimsplit model, support batch prediction.
        """
        assert self.claimsplit_model is not None and self.claimsplit_tokenizer is not None, "claimsplit model is not loaded"
        sentences = [sentences] if isinstance(sentences, str) else sentences
        all_sentences_batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
        all_predictions = []
        if verbose:
            pbar = tqdm(total=len(all_sentences_batches))
            pbar.set_description(f'splitting sentences into sub-claims')
        for sentences in all_sentences_batches:
            sentences = [task_prefix + item.strip() for item in sentences]
            inputs = self.claimsplit_tokenizer(
                    sentences, padding='longest', truncation=True, max_length=max_source_length, return_tensors='pt'
                ).to(self.claimsplit_model.device)
            outputs = self.claimsplit_model.generate(**inputs, max_new_tokens=max_target_length)
            predictions = self.claimsplit_tokenizer.batch_decode(outputs, skip_special_tokens=False)
            sep_token, pad_token, eos_token = self.claimsplit_tokenizer.sep_token, self.claimsplit_tokenizer.pad_token, self.claimsplit_tokenizer.eos_token
            predictions = [[pred.strip() for pred in preds.strip(pad_token).strip(eos_token).split(sep_token)] for preds in predictions]
            # fix potential repetition of claimsplit model as well as truncation error when the input sequence is too long
            predictions = [[pred for i, pred in enumerate(preds) if pred not in ' '.join(preds[:i])] for preds in predictions] 
            
            all_predictions.extend(predictions)

            if verbose:
                pbar.update(1)

        return all_predictions if not isinstance(sentences, str) else all_predictions[0]

    def _parse_summary(
        self, 
        summary: Union[str, List[str]],
        claimsplit: bool = False,
        max_source_length=64,
        max_target_length=128,
        batch_size=256,
        verbose=False
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Given a (list of) summary, parse it (them) into mappings of `sentence` to their `citations`; 
        
        If `claimsplit` is True, will also split each sentence into a list of claims.

        Args:
            summary: 
                the summary text with citations or a list of summaries with citations
            claimsplit:
                whether to split each sentence into a list of claims using the claimsplit model.
                If `self.claimsplit_model` is None, will take the whole sentence as its claim.
            max_source_length: 
                max sequence length of the input text for claimsplit, sequence longer than this will be truncated
            max_target_length: 
                max sequence length of the output text for claimsplit, sequence longer than this will be truncated
            batch_size: 
                number of sentences to infer at a time
            verbose:
                whether to show progress bar
        Return:
            if `summary` is a single summary, will return a single `summary_parsing`; if `summary` is a list of summarries, will return a list of `summary_parsing`. 

            `summary_parsing` is a list of dict where each item is: 
                {
                    'sentence': 
                        a sentence in the summary,
                    'citations': 
                        the set of citations for the senstence present in the summary,
                    'claims': 
                        the list of claims for the sentence if `claimsplit` is True
                }
        """
        squeeze = False
        if isinstance(summary, str):
            summary = [summary]
            squeeze = True
        all_summary_parsing = [parse_text_with_citations(item) for item in summary]
        if not claimsplit:
            return all_summary_parsing if not isinstance(summary, str) else all_summary_parsing[0]
        # claim-split
        all_summary_sen_nums = [len(item) for item in all_summary_parsing]
        all_sentences = [item['sentence'] for parsing in all_summary_parsing for item in parsing]
        if self.claimsplit_model is None: # if claimsplit model is not loaded, take the whole sentence as its claim
            all_claims = [[sen] for sen in all_sentences]
        all_claims = self._split_claims(
            sentences=all_sentences, 
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            batch_size=batch_size,
            verbose=verbose
        )
        # split predictions for each summary
        claims_for_all_summaries = [all_claims[i:i+l] for i, l in zip(np.cumsum([0]+all_summary_sen_nums)[:-1], all_summary_sen_nums)]
        for idx, claims_for_summary in enumerate(claims_for_all_summaries):
            summary_parsing = all_summary_parsing[idx]
            if summary_parsing == []:
                continue
            for i, item in enumerate(summary_parsing):
                item['claims'] = claims_for_summary[i]
            all_summary_parsing[idx] = summary_parsing
        if squeeze:
            all_summary_parsing = all_summary_parsing[0]
        return all_summary_parsing

    def _reduce_claims(
        self, 
        claims: Union[List[str], List[List[str]]],
        max_length=128,
        batch_size=256,
        reduction = 'single',
        verbose=False
    ) -> Union[List[List[str]], List[List[List[str]]]]:
        """
        First divide claims into clusters using the nli_model, where claims in the same cluster are redundant (entail each other),

        then reduce each cluster of claims into a single claim to eliminate redundancy.

        Support batch prediction.

        Args:
            claims: 
                the claims to be reduced, can be a list of claims or a list of list of claims (batch prediction)
            max_length:
                max sequence length of the input text, sequence longer than this will be truncated
            batch_size:
                number of <premise, hypothesis> pairs for the nli model to infer at a time
            reduction:
                the method to reduce claims in each cluster, can be 'single' or 'concat'.
                If 'single', will take the first claim in the cluster as the reduced claim;
                If 'concat', will concatenate all claims in the cluster into a single claim.
        """    
        assert self.nli_model is not None and self.nli_tokenizer is not None, "nli model is not loaded"
        squeeze = False
        if isinstance(claims[0], str):
            claims = [claims]
            squeeze = True
        all_claim_nums = [len(item) for item in claims]

        # prepare all <premise, hypothesis> pairs
        premises, hypotheses = [], []
        for i, claim_num in enumerate(all_claim_nums):
            for j in range(claim_num):
                for k in range(claim_num):
                    if j != k:
                        premises.append(claims[i][j])
                        hypotheses.append(claims[i][k])

        # run entailment
        # split into batches
        hypothesis_batches = [hypotheses[i:i+batch_size] for i in range(0, len(hypotheses), batch_size)]
        premise_batches = [premises[i:i+batch_size] for i in range(0, len(premises), batch_size)]
        # batch forward
        if verbose:
            pbar = tqdm(total=len(hypothesis_batches))
            pbar.set_description(f'reducing claims')
        all_predictions = []
        for hypothesis_batch, premise_batch in zip(hypothesis_batches, premise_batches):
            predictions = self._predict_nli(
                hypothesis=hypothesis_batch, premise=premise_batch, max_length=max_length
            )
            all_predictions.extend(predictions)

            if verbose:
                pbar.update(1)

        all_predictions = [self.id2label[pred]==self.entail_label for pred in all_predictions]

        # Divide claims into clusters, where claims in the same cluster entail each other
        all_clusters = []
        for i, claim_num in enumerate(all_claim_nums):
            claims_for_sample = claims[i]
            if claim_num == 0:
                all_clusters.append([])
                continue
            elif claim_num == 1:
                all_clusters.append([claims[i]])
                continue
            # make the entailment matrix
            entailment_matrix = np.zeros((claim_num, claim_num))
            for j in range(claim_num):
                for k in range(claim_num):
                    if j != k:
                        entailment_matrix[j][k] = all_predictions.pop(0)
                    else:
                        entailment_matrix[j][k] = 1
            clusters = []
            for j in range(claim_num):
                found_cluster = False
                for cluster in clusters:
                    if any(entailment_matrix[j][k] == 1 and entailment_matrix[k][j] == 1 for k in cluster):
                        cluster.append(j)
                        found_cluster = True
                        break
                if not found_cluster:
                    clusters.append([j])
            clusters = [[claims_for_sample[k] for k in cluster] for cluster in clusters]
            all_clusters.append(clusters)
        # reduce each cluster of claims into a single claim
        if reduction == 'single':
            all_clusters = [[cluster[0] for cluster in clusters] for clusters in all_clusters]
        elif reduction == 'concat':
            all_clusters = [[''.join(cluster) for cluster in clusters] for clusters in all_clusters]

        if squeeze:
            all_clusters = all_clusters[0]
        return all_clusters

    def evaluate_claimsplit(
        self,
        predictions: Union[List[str], List[List[str]]],
        references: Union[str, List[str]],
        max_seq_length=512,
        batch_size=256,
        verbose=False
    ) -> Dict[str, float]:
        """
        Evaluate the claim-split model predictions using four metrics:

            num_splits: the number of reduced claims per reference (source sentence)

            redundancy: 1 - number of reduced claims / number of claims

            correctness: the fraction of claims that are entailed by the reference (source sentence)

            completeness: whether the reference (source sentence) is entailed by all claims

        Args:
            predictions: 
                The predictions of the claimsplit model, can be a list of claims (from one sentence)
                or a list of list of claims (from multiple sentences respectively)
            references: 
                The source sentence(s). If `predictions` is a list of claims, `references` should be a single sentence;
                If `predictions` is a list of list of claims, `references` should be a list of sentences.
            max_seq_length:
                max sequence length of the input text, sequence longer than this will be truncated
            batch_size:
                number of <premise, hypothesis> pairs for the nli model to infer at a time
            verbose:
                whether to show progress bar
        Return:
            a dict of metrics
        """
        if isinstance(references, str):
            assert isinstance(predictions[0], str), "the number of predictions is not equal to the number of references"
            predictions = [predictions]
            references = [references]
        else:
            assert len(predictions)==len(references), "the number of predictions is not equal to the number of references"
        # reduce claims
        all_reduced_claims = self._reduce_claims(
            claims=predictions, max_length=max_seq_length, batch_size=batch_size, verbose=verbose
        )
        num_splits = [len(reduced_claims) for reduced_claims in all_reduced_claims]
        redundancy_scores = [1 - len(reduced_claims) / len(predictions[i]) for i, reduced_claims in enumerate(all_reduced_claims)]
        # compute correctness and completeness
        premises, hypotheses = [], []
        all_sample_nli_input_nums = []
        for i, (prediction, reference) in enumerate(zip(predictions, references)):
            # input for correctness
            premises.extend([reference]*len(prediction))
            hypotheses.extend(prediction)
            # input for completeness
            premises.append(' '.join(prediction))
            hypotheses.append(reference)
            all_sample_nli_input_nums.append(len(prediction)+1)

        # run entailment
        # split into batches
        hypothesis_batches = [hypotheses[i:i+batch_size] for i in range(0, len(hypotheses), batch_size)]
        premise_batches = [premises[i:i+batch_size] for i in range(0, len(premises), batch_size)]
        if verbose:
            pbar = tqdm(total=len(hypothesis_batches))
            pbar.set_description(f'evaluating claimsplit predictions')
        all_nli_predictions = []
        for hypothesis_batch, premise_batch in zip(hypothesis_batches, premise_batches):
            nli_preds = self._predict_nli(
                hypothesis=hypothesis_batch, premise=premise_batch, max_length=max_seq_length
            )
            all_nli_predictions.extend(nli_preds)
            if verbose:
                pbar.update(1)
        all_nli_predictions = [self.id2label[pred]==self.entail_label for pred in all_nli_predictions]

        correctness_scores = []
        completeness_scores = []
        for i, (prediction, reference) in enumerate(zip(predictions, references)):
            # get nli_predictions_for_sample for each sample using np.cumsum
            nli_predictions_for_sample = all_nli_predictions[np.cumsum([0]+all_sample_nli_input_nums)[:-1][i]:np.cumsum([0]+all_sample_nli_input_nums)[1:][i]]
            assert len(nli_predictions_for_sample) == len(prediction)+1
            # compute correctness
            nli_preds = nli_predictions_for_sample[:-1]
            correctness = sum(nli_preds) / len(nli_preds)
            # compute completeness
            completeness = nli_predictions_for_sample[-1]
            correctness_scores.append(correctness)
            completeness_scores.append(completeness)
        
        scores = {
            'num_splits': np.mean(num_splits),
            'redundancy': np.mean(redundancy_scores),
            'correctness': np.mean(correctness_scores),
            'completeness': np.mean(completeness_scores)
        }
        return scores

    def _compute_claim_score(
        self,
        claims: Union[List[str], List[List[str]]],
        summary: Union[str, List[str]],
        max_seq_length=512,
        batch_size=256,
        verbose=False,
        metric_key_prefix: str=''
    ):
        """
        Compute the fraction of claims being entailed by the summary using the nli_model
        """
        squeeze = False
        if isinstance(claims[0], str):
            all_claims = [claims]
            all_summaries = [summary]
            squeeze = True
        else:
            all_claims = claims
            all_summaries = summary
        assert len(all_claims) == len(all_summaries)
        # prepare all <premise, hypothesis> pairs
        premises, hypotheses = [], []
        all_sample_nli_input_nums = []
        for i, (claims, summary) in enumerate(zip(all_claims, all_summaries)):
            for claim in claims:
                premises.append(summary)
                hypotheses.append(claim)
            all_sample_nli_input_nums.append(len(claims))
        
        # run entailment
        # split into batches
        hypothesis_batches = [hypotheses[i:i+batch_size] for i in range(0, len(hypotheses), batch_size)]
        premise_batches = [premises[i:i+batch_size] for i in range(0, len(premises), batch_size)]
        if verbose:
            pbar = tqdm(total=len(hypothesis_batches))
            pbar.set_description(f'computing {metric_key_prefix} claim score')
        all_nli_predictions = []
        for hypothesis_batch, premise_batch in zip(hypothesis_batches, premise_batches):
            nli_preds = self._predict_nli(
                hypothesis=hypothesis_batch, premise=premise_batch, max_length=max_seq_length
            )
            all_nli_predictions.extend(nli_preds)

            if verbose:
                pbar.update(1)
        all_nli_predictions = [self.id2label[pred]==self.entail_label for pred in all_nli_predictions]

        all_scores = []
        for i, (claims, summary) in enumerate(zip(all_claims, all_summaries)):
            # get nli_predictions_for_sample for each sample using np.cumsum
            nli_predictions_for_sample = all_nli_predictions[np.cumsum([0]+all_sample_nli_input_nums)[:-1][i]:np.cumsum([0]+all_sample_nli_input_nums)[1:][i]]
            assert len(nli_predictions_for_sample) == len(claims)
            matched_num = sum(nli_predictions_for_sample)
            all_scores.append(matched_num / len(claims) if len(claims) != 0 else 0.)

        if squeeze: 
            all_scores = all_scores[0]

        return all_scores
           
    def evaluate_summary(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]],
        claimsplit: bool = True,
        claimsplit_max_source_length=64,
        claimsplit_max_target_length=128,
        claimsplit_batch_size=128,
        nli_max_seq_length=512,
        nli_batch_size=256,
        verbose=False,
        **kwargs
    ):
        """
        Evaluate summarization utility by computing claim precision and claim recall
        """
        squeeze = False
        if isinstance(predictions, str):
            predictions = [predictions]
            squeeze = True
        if isinstance(references, str):
            references = [references]
        if isinstance(references[0], list):
            references = [' '.join(ref) for ref in references] # if multiple references for one prediction, concatenate them

        if claimsplit and self.claimsplit_model is None:
            claimsplit = False
            logger.warning("claimsplit model is not loaded, will not split claims")

        # do batch parsing (and claimplit) for all predictions and references together
        all_parsings = self._parse_summary(
                        predictions + references, 
                        claimsplit=claimsplit, 
                        batch_size=claimsplit_batch_size,
                        max_source_length=claimsplit_max_source_length,
                        max_target_length=claimsplit_max_target_length,
                        verbose=verbose,
                        **kwargs
                    )
        all_preds_claims = [] # [[ claims for prediction1 ], [ claims for prediction2 ], ...]
        for parsing in all_parsings[:len(predictions)]:
            if claimsplit:
                all_preds_claims.append([cl for item in parsing for cl in item['claims']])
            else:
                all_preds_claims.append([item['sentence'] for item in parsing])

        all_refs_claims = [] # [[ claims for reference1 ], [ claims for reference2 ], ...]
        for parsing in all_parsings[len(predictions):]:
            if claimsplit:
                all_refs_claims.append([cl for item in parsing for cl in item['claims']])
            else:
                all_refs_claims.append([item['sentence'] for item in parsing])

        scores = {}
        scores['claim_precision'] = self._compute_claim_score(
            claims=all_preds_claims, 
            summary=references, 
            max_seq_length=nli_max_seq_length, 
            batch_size=nli_batch_size, 
            verbose=verbose,
            metric_key_prefix='claim_precision'
        )
        scores['claim_recall'] = self._compute_claim_score(
            claims=all_refs_claims, 
            summary=predictions, 
            max_seq_length=nli_max_seq_length, 
            batch_size=nli_batch_size, 
            verbose=verbose,
            metric_key_prefix='claim_recall'
        )
        if squeeze:
            scores = {k: v[0] for k, v in scores.items()}

        return scores

    def _get_task_prefix_for_citation_mask(
        self,
        query: str = None,
    ) -> str:
        """
        Return the task prefix for citation_mask prediction.
        """
        if query is not None:
            prefix = f"查询词：{query}，搜索结果摘要："
        else:
            prefix = "搜索结果摘要："
        return prefix
    
    def _predict_citation(
        self, 
        summary_parsing: Union[List[Dict], List[List[Dict]]],
        docs: Union[List[str], List[List[str]]], 
        predict_citation_mask: bool = True,
        query: Union[str, List[str]] = None,
        citation_pred_key: Optional[str] = 'nearest_citations',
        citation_ref_key: Optional[str] = 'auto_citations',
        max_seq_length=512,
        batch_size=256,
        verbose=False
    ) -> List[List[Dict]]:
        """
        Given the summary_parsing and source docs, first predict the citation_mask for each sentence,

        then predict the set of citations for each sentence. Both tasks are formulated as NLI task on pairs of <premise, hypothesis>.

        A doc is predicted as a citation of a sentence if:
            1. the doc entails the sentence
            2. the doc does not contradict the sentence and entails at least one of its claims

        If 'claims' of sentences are not given in `summary_parsing`, will take the sentence itself as its claim.
        
        The citation index starts from 1, so the index of the first doc is 1, the index of the second doc is 2, and so on.

        Args:
            summary_parsing: 
                the parsing of a summary returned by `parse_summary`, or a list of parsings with respect to multiple summaries
            docs: 
                if 'summary_parsing' is the parsing of a single summary, `docs` should be a list of source docs;
                elif 'summary_parsing' is a list of parsings of multiple summaries, `docs` should be a list of list of source docs.
            predict_citation_mask:
                whether to predict the citation_mask for each sentence using the nli_model, will be ignored if `citation_mask` is given in `summary_parsing`.
                if set to `True` and 'citation_mask' is not given in `summmary_parsing`, will predict the citation_mask for each sentence: citation_mask is 1 if a sentence needs citation, else 0.
                if set to `False` and 'citation_mask' is not given in `summmary_parsing`, will assign citation_mask=1 to all sentences, indicating that all sentences need citation.
            query:
                if provided, will be added into the prompt to improve the accuracy of citation_mask prediction if `predict_citation_mask` is True;
                if 'summary_parsing' is the parsing of a single summary, `query` should be a single string;
                elif 'summary_parsing' is a list of parsings of multiple summaries, `query` should be a list of strings.
            citation_pred_key:
                the key to store the given (model-generated) citations for each sentence in `summary_parsing`
            citation_ref_key:
                the key to store the citations predicted by the evaluator for each sentence in `summary_parsing`
            max_seq_length: 
                max sequence length of the input text, sequence longer than this will be truncated
            batch_size: 
                number of <premise, hypothesis> pairs to infer at a time
            verbose:
                whether to show progress bar

        Return:
            updated 'summary_parsing' with the following fields added:

                'citation_mask': 1 if the sentence needs citation, else 0,

                'ais': 1 if the sentence is fully supported by the given citations, else 0,

                'acs': 1 if the sentence is fully supported by the evaluator-predicted citations, else 0,

                'claim2rels': the mapping of each claim to its relation with each doc. 
        """
        all_hypotheses, all_premises = [], []
        all_claim_nums = []
        # input check
        squeeze = False
        if isinstance(summary_parsing[0], dict):
            assert isinstance(docs[0], str), "`docs` should be a list of source strings"
            if query is not None:
                assert isinstance(query, str), "`query` should be a single string"
                all_queries = [query]
            else:
                all_queries = [ModuleNotFoundError]
            all_summary_parsing = [summary_parsing]
            all_docs = [docs]
            squeeze = True
        else:
            all_summary_parsing = summary_parsing
            all_docs = docs
            assert isinstance(docs[0], list), "`docs` should be a list of list of source strings"
            assert len(summary_parsing) == len(docs)
            if query is not None:
                assert isinstance(query, list), "`query` should be a list of strings"
                all_queries = query
                assert len(summary_parsing) == len(all_queries)
            else:
                all_queries = [None]*len(all_summary_parsing)
            
        # prepare all <premise, hypothesis> pairs
        for i, summary_parsing in enumerate(all_summary_parsing):
            claim_nums_of_sens = []
            for item in summary_parsing:
                # we will add the sentence itself into `claims`, since we also need to predict the relation bewteen the sentence and the doc
                if 'claims' not in item or len(item['claims']) <= 1:
                    item['claims'] = [item['sentence'].strip()]
                if item['claims'][-1].strip() != item['sentence'].strip():
                    item['claims'].append(item['sentence'].strip())
                claim_nums_of_sens.append(len(item['claims']))
            all_claim_nums.append(claim_nums_of_sens)

            sumamry_hypotheses, summary_premises = [], []
            for item in summary_parsing:
                # first, predict wether each sentence needs citation
                # a sentence does not need citation only of all of its claims are supported by other sentences with citations in the same summary
                prefix_for_citation_mask = self._get_task_prefix_for_citation_mask(all_queries[i])
                citation_mask_premise = ''.join([d['sentence'] for d in summary_parsing if d['sentence']!=item['sentence'] and d['citations']]) # concatenate all other sentences with non-empty citations as premise
                citation_mask_premise = prefix_for_citation_mask + citation_mask_premise
                for claim in item['claims']:
                    hypothesis = claim.strip()
                    sumamry_hypotheses.append(hypothesis)
                    summary_premises.append(citation_mask_premise)
                    # second, predict the relation for each claim and document
                    for doc in all_docs[i]:
                        doc_premise = doc.strip()
                        sumamry_hypotheses.append(hypothesis)
                        summary_premises.append(doc_premise)

            # gather all <premise, hypothesis> pairs
            all_hypotheses.append(sumamry_hypotheses)
            all_premises.append(summary_premises)

        # split data to batches with size `batch_size`
        all_hypotheses_flattened = [h for hypotheses in all_hypotheses for h in hypotheses]
        all_premises_flattened = [p for premises in all_premises for p in premises]
        hypothesis_batches = [all_hypotheses_flattened[i:i+batch_size] for i in range(0, len(all_hypotheses_flattened), batch_size)]
        premise_batches = [all_premises_flattened[i:i+batch_size] for i in range(0, len(all_premises_flattened), batch_size)]
        # batch forward
        all_predictions = []
        if verbose:
            pbar = tqdm(total=len(hypothesis_batches))
            pbar.set_description(f'predicting citations')
        for hypothesis, premise in zip(hypothesis_batches, premise_batches):
            predictions = self._predict_nli(
                hypothesis=hypothesis, premise=premise, max_length=max_seq_length
            )
            all_predictions.extend(predictions)
            if verbose:
                pbar.update(1)

        # split predictions for each summary
        all_summaries_pred_nums = [len(hypotheses) for hypotheses in all_hypotheses]
        predictions_for_all_summaries = [all_predictions[i:i+l] for i, l in zip(np.cumsum([0]+all_summaries_pred_nums)[:-1], all_summaries_pred_nums)]

        for idx, predictions_for_summary in enumerate(predictions_for_all_summaries):
            summary_parsing = all_summary_parsing[idx]
            docs = all_docs[idx]
            num_docs = len(docs)
            if summary_parsing == []:
                continue
            # split predictions on claims for each sentence
            claim_num_for_sentences = all_claim_nums[idx]
            assert len(claim_num_for_sentences) == len(summary_parsing) # number of sentences in the summary
            all_sentences_pred_nums = [(1+num_docs)*claim_num for claim_num in claim_num_for_sentences]
            predictions_for_sentences = [predictions_for_summary[i:i+l] for i, l in zip(np.cumsum([0]+all_sentences_pred_nums)[:-1], all_sentences_pred_nums)]
            for i, preds_for_sen in enumerate(predictions_for_sentences):
                num_claims = claim_num_for_sentences[i]
                preds_for_claims = np.array(preds_for_sen).reshape(num_claims, 1+num_docs)
                citation_mask_for_sen = 1  # first assume that the sentence needs citation
                claims_doc_relations = [[] for _ in range(num_claims)]
                for j, preds in enumerate(preds_for_claims):
                    citation_mask_pred, relation_preds = preds[0], preds[1:]
                    if predict_citation_mask and self.id2label[citation_mask_pred] == self.entail_label:
                            # if the sentence is entailed by other sentences with citations
                            citation_mask_for_sen = 0
                    # identify the relation of each claim with each doc
                    for k, doc in enumerate(docs):
                        claims_doc_relations[j].append(self.id2label[relation_preds[k]])
                        
                claim2rels = {claim: rels for claim, rels in zip(summary_parsing[i]['claims'], claims_doc_relations)}
                summary_parsing[i]['claim2rels'] = claim2rels
                if 'citation_mask' not in summary_parsing[i].keys():
                    if summary_parsing[i]['citations'] != []: # if citation set is not empty, citation_mask will be 1
                        summary_parsing[i]['citation_mask'] = 1
                    else:
                        summary_parsing[i]['citation_mask'] = citation_mask_for_sen

                # identify the relation of each sentence with each doc, then determine citations and attribution
                auto_citations = []
                for k, doc in enumerate(docs):
                    cite_idx = k + 1 # citation index starts from 1
                    rel2claims = [rels[k] for rels in claim2rels.values()]
                    # if the doc contradicts the sentence, it should not be cited
                    if rel2claims[-1] == self.contradict_label:
                        continue
                    # otherwise, if the doc entails any claim, it should be cited
                    elif self.entail_label in rel2claims:
                        auto_citations.append(cite_idx)     
                summary_parsing[i][citation_ref_key] = auto_citations

                if summary_parsing[i]['citation_mask']==0:
                    ais = 1 
                    acs = 1
                else:
                    # determine the AIS score with the given citations
                    ais = 0
                    cited_doc_indices = [citation-1 for citation in summary_parsing[i][citation_pred_key]] # citation index starts from 1
                    # sentence is not attributable if it contradicts any cited doc
                    if self.contradict_label in [rel for d_i, rel in enumerate(list(claim2rels.values())[-1]) if d_i in cited_doc_indices]:
                        ais = 0
                    # otherwise, sentence is attributable if it is entailed by the any cited doc
                    elif self.entail_label in [rel for d_i, rel in enumerate(list(claim2rels.values())[-1]) if d_i in cited_doc_indices]:
                        ais = 1
                    elif len(claim2rels) > 1: # or if all of its claims are entailed by the cited docs
                        all_claims_entailed = True
                        for rels in list(claim2rels.values())[:-1]:
                            if self.entail_label not in [rel for i, rel in enumerate(rels) if i in cited_doc_indices]:
                                all_claims_entailed = False
                                break
                        if all_claims_entailed:
                            ais = 1
                            
                    # determine the ACS score with the evaluator-predicted citations
                    acs = 0
                    evaluator_cited_doc_indices = [citation-1 for citation in summary_parsing[i][citation_ref_key]]
                    if self.contradict_label in [rel for _i, rel in enumerate(list(claim2rels.values())[-1]) if _i in evaluator_cited_doc_indices]:
                        acs = 0
                    elif self.entail_label in [rel for _i, rel in enumerate(list(claim2rels.values())[-1]) if _i in evaluator_cited_doc_indices]:
                        acs = 1
                    elif len(claim2rels) > 1:
                        all_claims_entailed = True
                        for rels in list(claim2rels.values())[:-1]:
                            if self.entail_label not in [rel for i, rel in enumerate(rels) if i in evaluator_cited_doc_indices]:
                                all_claims_entailed = False
                                break
                        if all_claims_entailed:
                            acs = 1
                
                summary_parsing[i]['ais'] = ais
                summary_parsing[i]['acs'] = acs

            all_summary_parsing[idx] = summary_parsing

        if squeeze:
            all_summary_parsing = all_summary_parsing[0]
        return all_summary_parsing
    
    def _compute_attribution_score(
        self, 
        summary_parsing: List[Dict],
        citation_pred_key: Optional[str] = 'nearest_citations',
        citation_ref_key: Optional[str] = 'auto_citations'
    ) -> Dict[str, float]:
        """
        Compute the citation score for a summary given the predicted citations for each sentence.

        """
        citation_p, citation_r, ais, acs = [], [], [], []
        for idx, item in enumerate(summary_parsing):
            if item['citation_mask']==0:
                continue
            citation_prediction = set(item[citation_pred_key]) 
            citation_reference = set(item[citation_ref_key])

            if len(citation_prediction) == 0:
                precision, recall = 0., 0.
                tp = []
            elif len(citation_reference) == 0:
                precision, recall = 0., 0.
                tp = []
            else:
                tp = citation_reference.intersection(citation_prediction)
                precision = len(tp) / len(citation_prediction)
                recall = len(tp) / len(citation_reference)

            citation_p.append(precision)
            citation_r.append(recall)
            ais.append(item['ais'])
            acs.append(item['acs'])

        # citation scores of the summary are the average of sentences
        citation_p = np.mean(citation_p) if citation_p else 0.
        citation_r = np.mean(citation_r) if citation_r else 0.
        ais = np.mean(ais) if ais else 0.
        acs = np.mean(acs) if acs else 0.
        score = {
            "citation_precision": round(citation_p, 4),
            "citation_recall": round(citation_r, 4),
            "ais": round(ais, 4),
            "acs": round(acs, 4)
        }
        return score

    def evaluate_attribution(
        self, 
        summary: Union[str, List[str]],
        docs: Union[List[str], List[List[str]]], 
        query: Optional[Union[str, List[str]]] = None,
        predict_citation_mask = True,
        citation_pred_key: Optional[str] = 'nearest_citations',
        citation_ref_key: Optional[str] = 'auto_citations',
        claimsplit: bool = True,
        claimsplit_max_source_length=64,
        claimsplit_max_target_length=128,
        claimsplit_batch_size=256,
        nli_max_seq_length=512,
        nli_batch_size=256,
        return_score_only=False,
        verbose=False,
    ) -> Union[List, Dict]:
        """
        Given the (a list of) summary and source docs, automatically predict the citations for each sentence in the summary and evaluate attribution.

        Support batch evaluation.

        The citation index starts from 1, so the index of the first doc is 1, the index of the second doc is 2, and so on.
        
        Args:
            summary:
                a single summary or a list of summaries to evaluate
            docs:
                if `summary` is a single summary, `docs` should be a list of source docs;
                elif `summary` is a list of summaries, `docs` should be a list of list of source docs. 
            query: 
                if provided, will add query into the premise prompt for citation_mask prediction, might improve accuracy by providing topic information;
                if `summary` is a single summary, `query` should be a single string;
                if `summary` is a list of summaries, `query` should be a list of strings.
            predict_citation_mask:
                whether to predict the citation_mask for each sentence using the nli_model, will be ignored if `citation_mask` is given in `summary_parsing`.
                if set to `True` and 'citation_mask' is not given in `summmary_parsing`, will predict the citation_mask for each sentence: citation_mask is 1 if a sentence needs citation, else 0.
                if set to `False` and 'citation_mask' is not given in `summmary_parsing`, will assign citation_mask=1 to all sentences, indicating that all sentences need citation.
            citation_pred_key:
                the key to store the given (model-generated) citations for each sentence in `summary_parsing`
            citation_ref_key:
                the key to store the citations predicted by the evaluator for each sentence in `summary_parsing`
            claimsplit:
                whether to split each sentence into a list of claims using the claimsplit model, will be set to `False` if `claimsplit_model` is not loaded
            claimsplit_max_source_length: 
                max sequence length of the input text for claimsplit, sequence longer than this will be truncated
            claimsplit_max_target_length: 
                max sequence length of the output text for claimsplit, sequence longer than this will be truncated
            claimsplit_batch_size:
                number of sentences for the claimsplit model to infer at a time
            max_seq_length: 
                max sequence length of the input text, sequence longer than this will be truncated
            batch_size: 
                number of <premise, hypothesis> pairs to infer at a time
            return_score_only:
                if set to `True`, return only the citation scores 
                otherwise, will return the sumamry parsing as well
            verbose:
                whether to show progress bar
        """
        # input check
        squeeze = False
        if isinstance(summary, str):
            summary = [summary]
            docs = [docs]
            if query is not None:
                query = [query]
            squeeze = True
        assert len(summary) == len(docs)
        if query is not None:
            assert len(summary) == len(query)

        if claimsplit and self.claimsplit_model is None:
            claimsplit = False
            logger.warning("claimsplit model is not loaded, will not split claims")

        all_parsings = self._parse_summary(
                        summary=summary,
                        claimsplit=claimsplit,
                        max_source_length=claimsplit_max_source_length,
                        max_target_length=claimsplit_max_target_length,
                        batch_size=claimsplit_batch_size,
                        verbose=verbose
                    )
        all_parsings = self._predict_citation(
                        summary_parsing=all_parsings, 
                        docs=docs,                                 
                        predict_citation_mask=predict_citation_mask,
                        citation_pred_key=citation_pred_key,
                        citation_ref_key=citation_ref_key,
                        query=query,
                        max_seq_length=nli_max_seq_length, 
                        batch_size=nli_batch_size,
                        verbose=verbose
                    )
        all_scores = [self._compute_attribution_score(summary_parsing, citation_pred_key, citation_ref_key) for summary_parsing in all_parsings]
        metric_names = all_scores[0].keys()
        all_scores = {metric_name: [score[metric_name] for score in all_scores] for metric_name in metric_names}

        if squeeze:
            all_scores = {k: v[0] for k, v in all_scores.items()}
            all_parsings = all_parsings[0]

        if return_score_only:
            return all_scores
        else:
            return all_scores, all_parsings

