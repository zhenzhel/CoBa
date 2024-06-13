import warnings
from typing import TYPE_CHECKING, Union
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers import GenerationConfig, PhrasalConstraint, DisjunctiveConstraint
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers import MaxLengthCriteria
import copy
import inspect

if TYPE_CHECKING:
    from transformers import BaseStreamer

def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = copy.deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria

logger = logging.get_logger(__name__)

from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
)
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]

class CoBaGenerator(GenerationMixin):
    def __init__(self, model, 
                 tokenizer,
                 use_coba,
                 coba_prob_thresh, 
                 coba_include_dist, 
                 coba_dist_thresh,
                 adjust_by_unconditional,
                 adjust_by_unconditional_scale,
                 coba_max_steps=2000,
                 use_batch=False,
                 llama=False,
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.main_input_name = model.main_input_name
        self.use_coba = use_coba
        self.coba_prob_thresh = coba_prob_thresh
        self.coba_include_dist = coba_include_dist
        self.coba_dist_thresh = coba_dist_thresh
        self.adjust_by_unconditional = adjust_by_unconditional
        self.adjust_by_unconditional_scale = adjust_by_unconditional_scale
        self.inputs_uncond = None
        self.attention_mask_uncond = None
        
        self.context_embedding_matrix = None
        self.context_embedding_matrix_norm = None
        if llama:
            self._embedding = self.model.get_input_embeddings().weight.detach().clone()
        else:
            self._embedding = self.model.shared.weight.detach().clone() # vocab_size x hidden_dim
        self._embedding_norm = torch.linalg.vector_norm(self._embedding, dim=1).flatten() # vocab_size 
        self._dist = None

        self.coba_max_steps = coba_max_steps
        self.llama = llama
        self._coba_steps = None
        
        self.use_batch = use_batch
    

    @torch.no_grad()
    def precompute_context_embeddings(self, context_ids_tensor): # context_ids is batch size x max context len
        context_ids = context_ids_tensor.detach().cpu().numpy().tolist() # 2d-list
        context_ids = [list(set(ids)) for ids in context_ids]
        self.context_embedding_matrix = [self._embedding[ids] for ids in context_ids] # list of 2d tensors
        self.context_embedding_matrix_norm = [self._embedding_norm[ids] for ids in context_ids] # list of 1d tensors
    
    def _dist_reset(self, batch_size):
        self._dist = [dict() for _ in range(batch_size)]
    
    def reset_context_embedding_matrix(self):
        self.context_embedding_matrix = None
        self.context_embedding_matrix_norm = None

    @torch.no_grad()
    def compute_tok_dist_to_context(self, tok_ids): # tok_id: batch size x 1
        dists = torch.zeros(tok_ids.shape[0]).cuda()
        for i in range(tok_ids.shape[0]):
            tok_id = tok_ids[i].item()
            if tok_id not in self._dist[i].keys():
                tok_embedding = self._embedding[tok_id].unsqueeze(0) # 1 x hidden_dim
                tok_embedding_norm = self._embedding_norm[tok_id]
                tok_dist = (1 - (self.context_embedding_matrix[i] * tok_embedding).sum(dim=1) / tok_embedding_norm / self.context_embedding_matrix_norm[i]).min().item() # context_len x hidden_dim
                self._dist[i][tok_id] = tok_dist
            tok_dist = self._dist[i][tok_id]
            dists[i] = tok_dist
        return dists
    
    def set_uncond_inputs(self, inputs_uncond, attention_mask_uncond=None):
        self.inputs_uncond = inputs_uncond
        self.attenion_mask_uncond = attention_mask_uncond

    def can_generate(self):
        return self.model.can_generate()
    
    def greedy_search(self, input_ids, 
                      logits_processor, 
                      stopping_criteria, 
                      max_length=None, 
                      pad_token_id=None, 
                      eos_token_id=None, 
                      output_attentions=None, 
                      output_hidden_states=None, 
                      output_scores=None, 
                      return_dict_in_generate=None, 
                      synced_gpus=False, 
                      streamer=None, 
                      uncond_dict=None,
                      **model_kwargs):
        if self.use_coba:
            if self.use_batch:
                decode_fn = self.greedy_search_coba_batchified
            else:
                assert input_ids.shape[0] == 1, "cannot have batch size > 1 for unbatchified version"
                decode_fn = self.greedy_search_coba_unbatchified
        else:
            decode_fn = self.greedy_search_regular
        return decode_fn(input_ids=input_ids, 
                         logits_processor=logits_processor, 
                         stopping_criteria=stopping_criteria, 
                         max_length=max_length, 
                         pad_token_id=pad_token_id, 
                         eos_token_id=eos_token_id, 
                         output_attentions=output_attentions, 
                         output_hidden_states=output_hidden_states, 
                         output_scores=output_scores,
                         return_dict_in_generate=return_dict_in_generate,
                         synced_gpus=synced_gpus, 
                         streamer=streamer, 
                         uncond_dict=uncond_dict,
                         **model_kwargs)

    def greedy_search_regular(self, input_ids, 
                      logits_processor, 
                      stopping_criteria, 
                      max_length, 
                      pad_token_id, 
                      eos_token_id, 
                      output_attentions, 
                      output_hidden_states, 
                      output_scores, 
                      return_dict_in_generate, 
                      synced_gpus, 
                      streamer,
                      uncond_dict,
                      **model_kwargs):

        # init values
        original_length = input_ids.shape[1]
        assert input_ids.shape[0] == 1
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            if self.llama:
                return input_ids[:, original_length:]
            else:
                return input_ids
    
    def greedy_search_coba_unbatchified(self, input_ids, 
                      logits_processor, 
                      stopping_criteria, 
                      max_length, 
                      pad_token_id, 
                      eos_token_id, 
                      output_attentions, 
                      output_hidden_states, 
                      output_scores, 
                      return_dict_in_generate, 
                      synced_gpus, 
                      streamer,
                      uncond_dict,
                      **model_kwargs):
        
        if self.llama:
            prompt = input_ids.clone().detach()
            input_ids = torch.tensor([[29901]]).to(input_ids.device)

        self._coba_steps = None
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        
        STATE_BACKTRACK = 1
        STATE_FORWARD = 2
        FILL_TOKEN = -float("Inf")

        probs_unmasked_adjust = []
        probs_masked_adjust = []

        if self.adjust_by_unconditional:
            input_ids_uncond = uncond_dict['input_ids_uncond']
            model_kwargs_uncond = uncond_dict['model_kwargs_uncond']

        assert input_ids.shape[0] == 1, 'does not support batch decoding yet'

        STATE = STATE_FORWARD # start with forward mode
        COBA_MAX_STEPS = self.coba_max_steps
        coba_steps = 0
        outputs = None
        while True:
        
            if coba_steps >= COBA_MAX_STEPS:
                this_peer_finished = True
                if not synced_gpus:
                    break
                continue

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            ################### BACKTRACK STATE ######################
            if STATE == STATE_BACKTRACK:
                coba_steps += 1
                # get the list to move down to, and extract the next candidate token
                # the assumption is that the previous candidate is already masked out
                proposed_next_token_prob, proposed_next_token = probs_masked_adjust[-1].max(dim=-1)
                prob_test = proposed_next_token_prob.item() >= self.coba_prob_thresh
                
                state_transition = True
                if input_ids.shape[-1] > 1:
                    if (not prob_test):                        
                        state_transition = False
                    
                    if state_transition and self.coba_include_dist:
                        proposed_next_token_dist = self.compute_tok_dist_to_context(proposed_next_token.unsqueeze(1))
                        dist_test = proposed_next_token_dist.item() <= self.coba_dist_thresh

                        # repropose only if you are far from the context, but are still above the prob threshold
                        while prob_test and (not dist_test):
                            probs_masked_adjust[-1][0,proposed_next_token.item()] = FILL_TOKEN
                            # propose an alternative token
                            proposed_next_token_prob, proposed_next_token = probs_masked_adjust[-1].max(dim=-1)
                            proposed_next_token_dist = self.compute_tok_dist_to_context(proposed_next_token.unsqueeze(1))
                            # update
                            prob_test = proposed_next_token_prob.item() >= self.coba_prob_thresh
                            dist_test = proposed_next_token_dist.item() <= self.coba_dist_thresh

                        # if you failed one of distance or prob test, and also you are not in context
                        if (not prob_test) or (not dist_test):
                            state_transition = False                    
                
                if not state_transition:
                    # stay in STATE_BACKTRACK
                    # move up the tree; remove every information in the current level
                    probs_masked_adjust = probs_masked_adjust[:-1]
                    probs_unmasked_adjust = probs_unmasked_adjust[:-1]

                    cur_last_token = input_ids[0, -1]
                    input_ids = input_ids[:,:-1]

                    if self.adjust_by_unconditional:
                        input_ids_uncond = input_ids_uncond[:,:-1]

                    assert probs_masked_adjust[-1].shape[0] == 1
                    probs_masked_adjust[-1][0,cur_last_token] = FILL_TOKEN # mask out the current candidate

                    if return_dict_in_generate:
                        if output_scores: 
                            scores = scores[:-1]
                        if output_attentions:
                            decoder_attentions = decoder_attentions[:-1]
                            if self.config.is_encoder_decoder:
                                cross_attentions = cross_attentions[:-1]
                        if output_hidden_states:
                            decoder_hidden_states = decoder_hidden_states[:-1]
                    continue

                else: 
                    next_tokens = proposed_next_token.unsqueeze(0)
                    if eos_token_id is not None:
                        if pad_token_id is None:
                            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)


                    # state transition to STATE_FORWARD
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    if self.adjust_by_unconditional:
                        input_ids_uncond = torch.cat([input_ids_uncond, next_tokens], dim=-1)

                    STATE = STATE_FORWARD # state transition
                    # finished sentences should have their next token be a padding token
                    
                    # if eos_token was found in one sentence, set sentence to finished
                    if eos_token_id_tensor is not None:
                        unfinished_sequences = unfinished_sequences.mul(
                            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                        )

                        # stop when each sentence is finished
                        if unfinished_sequences.max() == 0:
                            this_peer_finished = True

                    # stop if we exceed the maximum length
                    if stopping_criteria(input_ids, scores):
                        this_peer_finished = True

                    if this_peer_finished and not synced_gpus:
                        break

                    continue
            
            ################### FORWARD #######################
            else: # STATE == STATE_FORWARD
                assert STATE == STATE_FORWARD, 'state error'
                coba_steps += 1

                # prepare model inputs
                if self.llama:
                    prompt_input = torch.concat((prompt, input_ids), dim=1)
                    model_kwargs['attention_mask'] = torch.ones(1, prompt_input.shape[-1]).to(input_ids.device)
                    model_inputs = self.model.prepare_inputs_for_generation(prompt_input, **model_kwargs)
                else:
                    model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                if self.adjust_by_unconditional:
                    model_inputs_uncond = self.model.prepare_inputs_for_generation(input_ids_uncond, **model_kwargs_uncond)

                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                if self.adjust_by_unconditional:
                    outputs_uncond = self.model(
                        **model_inputs_uncond, 
                        return_dict=True,
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
                
                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need
                
                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                next_probs_unmasked_adjust = F.softmax(next_tokens_scores, dim=-1)

                if self.adjust_by_unconditional:
                    next_token_logits_uncond = outputs_uncond.logits[:, -1, :]
                    next_tokens_scores_uncond = logits_processor(input_ids_uncond, next_token_logits_uncond)
                    
                    next_tokens_scores_adjust = logits_processor(input_ids, 
                        (1+self.adjust_by_unconditional_scale)*next_token_logits-self.adjust_by_unconditional_scale*next_token_logits_uncond
                        )
                    next_probs_unmasked_adjust = F.softmax(next_tokens_scores_adjust, dim=-1)

                next_probs_masked_adjust = next_probs_unmasked_adjust.clone().detach()
                proposed_next_token_prob, proposed_next_token = next_probs_masked_adjust.max(dim=-1)
                prob_test = proposed_next_token_prob.item() >= self.coba_prob_thresh
                
                # CASE 1: hit hallucination criterion, backtrack immediately                
                state_transition = False
                if input_ids.shape[-1] > 1:
                    # check for token prob
                    if not prob_test:
                        state_transition = True
                    
                    # check for token dist
                    if (not state_transition) and self.coba_include_dist: 
                        proposed_next_token_dist = self.compute_tok_dist_to_context(proposed_next_token.unsqueeze(1)) 
                        dist_test = proposed_next_token_dist.item() <= self.coba_dist_thresh
                        # repropose only if you are far from the context, but are still above the prob threshold
                        while prob_test and (not dist_test):
                            next_probs_masked_adjust[0,proposed_next_token.item()] = FILL_TOKEN
                            # propose an alternative token
                            proposed_next_token_prob, proposed_next_token = next_probs_masked_adjust.max(dim=-1)
                            proposed_next_token_dist = self.compute_tok_dist_to_context(proposed_next_token.unsqueeze(1))
                            # update
                            prob_test = proposed_next_token_prob.item() >= self.coba_prob_thresh
                            dist_test = proposed_next_token_dist.item() <= self.coba_dist_thresh

                        # if you failed one of distance or prob test, and also you are not in context
                        if (not prob_test) or (not dist_test):
                            state_transition = True

                    if state_transition:
                        # mark the current last input as finished and abort
                        cur_last_token = input_ids[0,-1]
                        probs_masked_adjust[-1][:,cur_last_token] = FILL_TOKEN
                        input_ids = input_ids[:, :-1]

                        if self.adjust_by_unconditional:
                            input_ids_uncond = input_ids_uncond[:, :-1]

                        STATE = STATE_BACKTRACK
                        continue
                
                # no state transition
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_tokens_scores,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                
                next_tokens = proposed_next_token # torch.argmax(next_probs_masked_adjust, dim=-1)
                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                probs_masked_adjust.append(next_probs_masked_adjust)
                probs_unmasked_adjust.append(next_probs_unmasked_adjust)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if self.adjust_by_unconditional:
                    input_ids_uncond = torch.cat([input_ids_uncond, next_tokens[:, None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if self.adjust_by_unconditional:
                    model_kwargs_uncond = self._update_model_kwargs_for_generation(
                        outputs_uncond, model_kwargs_uncond, is_encoder_decoder=self.config.is_encoder_decoder
                    )
                

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                    # stop when each sentence is finished
                    if unfinished_sequences.max() == 0:
                        this_peer_finished = True

                # stop if we exceed the maximum length
                if stopping_criteria(input_ids, scores):
                    this_peer_finished = True

                if this_peer_finished and not synced_gpus:
                    break
        
        self._coba_steps = [coba_steps]

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
    def greedy_search_coba_batchified(self, input_ids, 
                      logits_processor, 
                      stopping_criteria, 
                      max_length, 
                      pad_token_id, 
                      eos_token_id, 
                      output_attentions, 
                      output_hidden_states, 
                      output_scores, 
                      return_dict_in_generate, 
                      synced_gpus, 
                      streamer,
                      uncond_dict,
                      **model_kwargs):
        
        original_length = input_ids.shape[1]
        if self.llama:
            prompt = input_ids.clone().detach()
            input_ids = torch.tensor([[29901]]).to(input_ids.device)
        self._coba_steps = None
        vocab_size = self.model.config.vocab_size
        batch_size = input_ids.shape[0]
        STATE_BACKTRACK = 0
        STATE_FORWARD = 1
        FILL_TOKEN = -float("Inf")
        COBA_MAX_STEPS = self.coba_max_steps

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only 

        probs_unmasked_adjust = torch.zeros(batch_size, 0, vocab_size).cuda()
        probs_masked_adjust = torch.zeros(batch_size, 0, vocab_size).cuda()

        if self.adjust_by_unconditional:
            input_ids_uncond = uncond_dict['input_ids_uncond'] # Should be batch_size x 1
            model_kwargs_uncond = uncond_dict['model_kwargs_uncond']
        
        # this is a mask where 1 = seqs in forward state; 0 = seqs in backtrack state
        STATE = torch.ones(batch_size, dtype=torch.long).fill_(STATE_FORWARD).cuda() # start with forward mode
        STATE_MASK = torch.ones(*input_ids.shape).cuda() # batch_size x max_seq_len_uptonow

        coba_steps = torch.zeros(batch_size).cuda()
        outputs = None

        while True:
            if unfinished_sequences.max() > 0 and coba_steps[unfinished_sequences.bool()].min() >= COBA_MAX_STEPS:
                this_peer_finished = True
                # print('exceeded limit')
                if not synced_gpus:
                    break
                continue

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            ################### FORWARD STATE #################
            # Update select column after updating STATE_MASK
            STATE_MASK_last_idx = (STATE_MASK.sum(dim=1)-1) # batch_size 
            STATE_MASK_last_idx_expand = STATE_MASK_last_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,vocab_size).long() # n_forward x 1 x vocab

            input_ids = (input_ids * STATE_MASK + pad_token_id * (1 - STATE_MASK)).type(input_ids.dtype) # batch_size x gen_len
            if self.llama:
                prompt_input = torch.concat((prompt, input_ids), dim=1)
                model_kwargs['attention_mask'] = torch.ones(1, prompt_input.shape[-1]).to(input_ids.device)
                model_inputs = self.model.prepare_inputs_for_generation(prompt_input, **model_kwargs)
            else:
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if self.adjust_by_unconditional:
                input_ids_uncond = (input_ids_uncond * STATE_MASK + pad_token_id * (1 - STATE_MASK)).type(input_ids_uncond.dtype)
                model_inputs_uncond = self.model.prepare_inputs_for_generation(input_ids_uncond, **model_kwargs_uncond)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_attentions,
            )
            if self.adjust_by_unconditional:
                outputs_uncond = self.model(
                    **model_inputs_uncond, 
                    return_dict=True,
                    output_attentions=output_attentions, 
                    output_hidden_states=output_hidden_states
                )
            
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            
            # outputs.logits: batch_size x gen_len x vocab_size
            # next_token_logits: batch_size x 1 x vocab_size --> batch_size x vocab_size
            if self.llama:
                next_token_logits = torch.gather(outputs.logits, dim=1, index=STATE_MASK_last_idx_expand + prompt.shape[-1]).squeeze(1)
            else:
                next_token_logits = torch.gather(outputs.logits, dim=1, index=STATE_MASK_last_idx_expand).squeeze(1)

            # pre-process distribution 
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_probs_unmasked = F.softmax(next_tokens_scores, dim=-1)

            next_probs_unmasked_adjust = next_probs_unmasked.clone().detach()

            if self.adjust_by_unconditional:
                next_token_logits_uncond = torch.gather(outputs_uncond.logits, dim=1, index=STATE_MASK_last_idx_expand).squeeze(1) # batch_size x vocab_size
                next_tokens_scores_uncond = logits_processor(input_ids_uncond, next_token_logits_uncond)
                
                next_tokens_logits_adjust = (1+self.adjust_by_unconditional_scale) * next_token_logits - self.adjust_by_unconditional_scale * next_token_logits_uncond
                next_tokens_scores_adjust = logits_processor(input_ids, next_tokens_logits_adjust)
                next_probs_unmasked_adjust = F.softmax(next_tokens_scores_adjust, dim=-1)

            next_probs_masked_adjust = next_probs_unmasked_adjust.clone().detach()
            proposed_next_token_score, next_tokens = torch.max(
                            next_probs_masked_adjust, dim=-1)
            # CASE 1: hit hallucination criterion, backtrack immediately
            # Only seqs that are currently in STATE_FORWARD and have more than 1 token now can backtrack
            f2b = torch.logical_and(STATE == STATE_FORWARD, (STATE_MASK.sum(dim=-1) > 1).flatten())
            # test backtracking criteria
            backtrack_test = proposed_next_token_score.flatten() < self.coba_prob_thresh 
            if self.coba_include_dist:
                assert next_tokens.shape[0] == 1, 'cannot handle batchified'
                dist = self.compute_tok_dist_to_context(next_tokens) 
                while dist[0].item() > self.coba_dist_thresh:
                    next_probs_masked_adjust[0,next_tokens[0].item()] = FILL_TOKEN
                    # if you already cannot select next token, just keep the current one
                    if next_probs_masked_adjust.max() < self.coba_prob_thresh:
                        break
                    # try another one
                    proposed_next_token_score, next_tokens = torch.max(
                            next_probs_masked_adjust, dim=-1)
                    dist = self.compute_tok_dist_to_context(next_tokens) 

                backtrack_test = proposed_next_token_score.flatten() < self.coba_prob_thresh 
                backtrack_test = torch.logical_or(backtrack_test, dist > self.coba_dist_thresh)

            ################################################
            f2b = torch.logical_and(f2b, backtrack_test)
            ################################################


            f2f = torch.logical_and(STATE == STATE_FORWARD, ~f2b)
            f2f = torch.logical_and(f2f, unfinished_sequences.bool())
            f2b = torch.logical_and(f2b, unfinished_sequences.bool())

            STATE[f2b] = STATE_BACKTRACK
            STATE[f2f] = STATE_FORWARD
            
            # For FORWARD->BACKTRACK sequences, index into the last index with STATE_MASK 1
            # And change that to 0
            # For the other sequences, keep it to be 1
            STATE_MASK.scatter_(1, 
                                STATE_MASK_last_idx.unsqueeze(1).long(), # batch_size x 1
                                (1. - f2b.float()).unsqueeze(1) # 0 if f2b; 1 otherwise
                                )
            
            # first take the last token of the input_ids for f2b seqs, 
            # then fill the corresponding token place in probs_masked_adjust as FILL_TOKEN
            if probs_masked_adjust.shape[1] > 0:
                # 1: take the last token of input_ids for f2b seqs
                cur_last_tokens = torch.gather(input_ids[f2b,:], dim=1, 
                                    index=STATE_MASK_last_idx.long()[f2b].unsqueeze(1)) # f2b x 1

                # 2. get the corresponding probs for the last tokens
                f2b_FILL = torch.gather(probs_masked_adjust[f2b,:,:], 1, STATE_MASK_last_idx_expand[f2b]-1).squeeze(1)
                f2b_FILL.scatter_(1, cur_last_tokens, 
                                  torch.ones(f2b.shape[0], 1).fill_(FILL_TOKEN).cuda()) # (b2b_size x vocab_size)

                probs_masked_adjust[f2b,:,:] = probs_masked_adjust[f2b,:,:].scatter(
                    1, STATE_MASK_last_idx_expand[f2b]-1,  f2b_FILL.unsqueeze(1))

            input_ids = (input_ids * STATE_MASK + pad_token_id * (1 - STATE_MASK)).type(input_ids.dtype) # batch_size x gen_len

            # no state transition
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens.flatten()
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            next_tokens = next_tokens * STATE + pad_token_id * (1 - STATE)
            next_tokens = next_tokens.unsqueeze(1)

            # check if we need to expand tensor shape
            if f2f.sum() > 0 and (STATE_MASK_last_idx[f2f].max().item() >= probs_masked_adjust.shape[1]):
                # batch size x seq length x vocab_size
                probs_masked_adjust = torch.cat(
                    [probs_masked_adjust, 
                     torch.zeros(batch_size, 1, vocab_size).cuda()],
                     dim=1
                )
                probs_unmasked_adjust = torch.cat(
                    [probs_unmasked_adjust, 
                     torch.zeros(batch_size, 1, vocab_size).cuda()], 
                     dim=1
                )
                STATE_MASK = torch.cat([STATE_MASK, torch.zeros(batch_size, 1).cuda()], dim=1)
                input_ids = torch.cat([input_ids, torch.zeros(batch_size, 1, dtype=input_ids.dtype).cuda()], dim=1)
                if self.adjust_by_unconditional:
                    input_ids_uncond = torch.cat([input_ids_uncond, torch.zeros(batch_size, 1, dtype=input_ids_uncond.dtype).cuda()], dim=1)

            # update generated ids, model inputs, and length for next step
            probs_masked_adjust[f2f,:,:] = probs_masked_adjust[f2f,:,:].scatter(
                dim=1, index=STATE_MASK_last_idx_expand[f2f,:,:], 
                src=next_probs_masked_adjust[f2f].unsqueeze(1) # batch size@f2f x 1 x vocab 
            )
            probs_unmasked_adjust[f2f,:,:] = probs_unmasked_adjust[f2f,:,:].scatter(
                dim=1, index=STATE_MASK_last_idx_expand[f2f,:,:],
                src=next_probs_masked_adjust[f2f].unsqueeze(1)
            )
            # append next tokens to input_ids for FORWARD->FORWARD sequences
            # plus1 because we want to add to the position AFTER the current last idx of MASK=1
            STATE_MASK_last_idx_plus1_2D = STATE_MASK_last_idx[f2f].unsqueeze(1).long()+1
            input_ids[f2f,:] = input_ids[f2f,:].scatter(
                dim=1, index=STATE_MASK_last_idx_plus1_2D, src=next_tokens[f2f,:])
            if self.adjust_by_unconditional:
                input_ids_uncond[f2f,:] = input_ids_uncond[f2f,:].scatter(
                    dim=1, index=STATE_MASK_last_idx_plus1_2D, src=next_tokens[f2f,:])

            STATE_MASK[f2f,:] = STATE_MASK[f2f,:].scatter(
                dim=1, index=STATE_MASK_last_idx_plus1_2D, src=torch.ones(f2f.shape[0], 1).cuda())

            
            if streamer is not None:
                streamer.put(next_tokens.flatten().cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if self.adjust_by_unconditional:
                model_kwargs_uncond = self._update_model_kwargs_for_generation(
                    outputs_uncond, model_kwargs_uncond, is_encoder_decoder=self.config.is_encoder_decoder
                )
                
            coba_steps[f2b] += 1
            coba_steps[f2f] += 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.flatten().tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break
            
            ############ Backtrack State (STATE_BACKTRACK) #################
            # get the list to move down to, and extract the next candidate token
            # the assumption is that the previous candidate is already masked out
            STATE_MASK_last_idx = (STATE_MASK.sum(dim=1)-1) # 1D tensor of length batch_size
            STATE_MASK_last_idx_expand = STATE_MASK_last_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,vocab_size).long()

            # minus one because the shape of probs_*_adjust is 1 shorter than STATE_MASK
            b_all = (STATE == STATE_BACKTRACK)
            if b_all.sum().item() > 0:
                backtrack_select_probs_masked_adjust_dummy = torch.gather(
                    probs_masked_adjust[b_all,:,:], 1, STATE_MASK_last_idx_expand[b_all]).squeeze(1) # bt x vocab
                backtrack_select_token_score_dummy, backtrack_select_tokens_dummy = torch.max(
                    backtrack_select_probs_masked_adjust_dummy, dim=-1) # bt 
                backtrack_select_tokens_dummy = backtrack_select_tokens_dummy.unsqueeze(1) # btx1

                backtrack_select_probs_masked_adjust = torch.zeros(batch_size, vocab_size).cuda()
                backtrack_select_probs_masked_adjust[b_all,:] = backtrack_select_probs_masked_adjust_dummy
                backtrack_select_tokens = torch.zeros(batch_size, 1).long().cuda()
                backtrack_select_tokens[b_all,:] = backtrack_select_tokens_dummy
                backtrack_select_token_score = torch.zeros(batch_size).cuda()
                backtrack_select_token_score[b_all] = backtrack_select_token_score_dummy

                b2b = torch.logical_and(STATE == STATE_BACKTRACK, (STATE_MASK.sum(dim=-1) > 1).flatten()) # 1-D tensor w/ length batch_size BOOL
                backtrack_test = backtrack_select_token_score < self.coba_prob_thresh
                if self.coba_include_dist:
                    assert backtrack_select_tokens.shape[0] == 1, 'cannot support batch'
                    if b_all.sum() > 0:
                        assert backtrack_select_tokens_dummy.shape == backtrack_select_tokens.shape, 'select tok shape error'
                        assert backtrack_select_probs_masked_adjust_dummy.shape == backtrack_select_probs_masked_adjust.shape, 'shape error'
                        dist = self.compute_tok_dist_to_context(backtrack_select_tokens_dummy.flatten()) 
                        while dist[0].item() > self.coba_dist_thresh:
                            backtrack_select_probs_masked_adjust_dummy[0,backtrack_select_tokens_dummy[0,0].item()] = FILL_TOKEN
                            probs_masked_adjust[0,int(STATE_MASK_last_idx[0].item()), backtrack_select_tokens_dummy[0,0].item()] = FILL_TOKEN
                            if backtrack_select_probs_masked_adjust_dummy.max() < self.coba_prob_thresh:
                                break
                            backtrack_select_token_score_dummy, backtrack_select_tokens_dummy = torch.max(
                                                                                backtrack_select_probs_masked_adjust_dummy, dim=-1) # bt 
                            backtrack_select_tokens_dummy = backtrack_select_tokens_dummy.unsqueeze(1)

                            backtrack_select_probs_masked_adjust = torch.zeros(batch_size, vocab_size).cuda()
                            backtrack_select_probs_masked_adjust[b_all,:] = backtrack_select_probs_masked_adjust_dummy
                            backtrack_select_tokens = torch.zeros(batch_size, 1).long().cuda()
                            backtrack_select_tokens[b_all,:] = backtrack_select_tokens_dummy
                            backtrack_select_token_score = torch.zeros(batch_size).cuda()
                            backtrack_select_token_score[b_all] = backtrack_select_token_score_dummy

                            dist = self.compute_tok_dist_to_context(backtrack_select_tokens_dummy.flatten()) 

                    backtrack_test = backtrack_select_token_score < self.coba_prob_thresh
                    backtrack_test = torch.logical_or(backtrack_test, dist > self.coba_dist_thresh)

                b2b = torch.logical_and(b2b, backtrack_test)
                ####

                b2f = torch.logical_and(STATE == STATE_BACKTRACK, ~b2b)
                b2f = torch.logical_and(b2f, unfinished_sequences.bool())
                b2b = torch.logical_and(b2b, unfinished_sequences.bool())

                STATE[b2b] = STATE_BACKTRACK
                STATE[b2f] = STATE_FORWARD
                # remove one for those still in backtrack state
                STATE_MASK.scatter_(dim=1, index=STATE_MASK_last_idx.unsqueeze(1).long(), 
                                    src=(1 - b2b.float()).unsqueeze(1))
                

                b2b_FILL_dummy = torch.gather(
                    probs_masked_adjust[b2b,:,:], 1, STATE_MASK_last_idx_expand[b2b]-1).squeeze(1) 
                b2b_FILL = torch.zeros(batch_size, vocab_size).cuda()
                b2b_FILL[b2b,:] = b2b_FILL_dummy
 
                b2b_MASK_TOKENS = torch.gather(input_ids[b2b,:], dim=1, index=STATE_MASK_last_idx[b2b].unsqueeze(1).long())
                b2b_FILL[b2b,:] = b2b_FILL[b2b,:].scatter(dim=1, index=b2b_MASK_TOKENS, 
                                        src=torch.ones(b2b.shape[0], 1).fill_(FILL_TOKEN).cuda()) # (b2b_size x vocab_size)
                # backtrack_select_probs_masked_adjust: batch size x vocab size --> batch size x 1 x vocab size
 
                probs_masked_adjust[b2b,:] = probs_masked_adjust[b2b,:].scatter(
                    dim=1, index=STATE_MASK_last_idx_expand[b2b,:,:]-1, # batch size x 1 x vocab size 
                    # batch size x 1 x vocab size
                    src=b2b_FILL[b2b,:].unsqueeze(1)) # batch size x seq_len x vocab size
                
                coba_steps[b2b] += 1
                coba_steps[b2f] += 1
                input_ids = (input_ids * STATE_MASK + pad_token_id * (1 - STATE_MASK)).type(input_ids.dtype) # batch_size x gen_len

                if return_dict_in_generate:
                    if output_scores: 
                        scores = scores[:-1]
                    if output_attentions:
                        decoder_attentions = decoder_attentions[:-1]
                        if self.config.is_encoder_decoder:
                            cross_attentions = cross_attentions[:-1]
                    if output_hidden_states:
                        decoder_hidden_states = decoder_hidden_states[:-1]


                # input_ids is batch_size x cur_max_seq_len
                if b2f.sum() > 0 and (STATE_MASK_last_idx[b2f].max()+1 >= input_ids.shape[1]):
                    input_ids = torch.cat([input_ids, torch.zeros(batch_size, 1, dtype=input_ids.dtype).cuda()], dim=1)
                    STATE_MASK = torch.cat([STATE_MASK, torch.zeros(STATE_MASK.shape[0], 1).cuda()], dim=1)
                    if self.adjust_by_unconditional:
                        input_ids_uncond = torch.cat([input_ids_uncond, torch.zeros(input_ids_uncond.shape[0], 1).cuda()], dim=1)

                # scatter
                STATE_MASK_last_idx_plus1_2D_b2f = STATE_MASK_last_idx[b2f].unsqueeze(1).long()+1

                input_ids[b2f,:] = input_ids[b2f,:].scatter(
                    dim=1, index=STATE_MASK_last_idx_plus1_2D_b2f, src=backtrack_select_tokens[b2f,:])
                STATE_MASK[b2f,:] = STATE_MASK[b2f,:].scatter(
                    dim=1, index=STATE_MASK_last_idx_plus1_2D_b2f, src=torch.ones(b2f.shape[0], 1).cuda()
                )
                if self.adjust_by_unconditional:
                    input_ids_uncond[b2f,:] = input_ids_uncond[b2f,:].scatter(
                    dim=1, index=STATE_MASK_last_idx_plus1_2D_b2f, src=backtrack_select_tokens[b2f,:]
                    )

        ############# after the while loop
        self._coba_steps = coba_steps.detach().cpu().numpy().tolist()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
    
    def sample(self, input_ids, logits_processor, 
               stopping_criteria, 
               max_length = None,
                pad_token_id = None,
                eos_token_id = None,
                output_attentions = None,
                output_hidden_states = None,
                output_scores = None,
                return_dict_in_generate = None,
                synced_gpus = False,
                streamer = None,
                uncond_dict=None,
                **model_kwargs,
            ):
        if self.use_coba:
            assert input_ids.shape[0] == 1, 'batch size must be 1'
            decode_fn = self.sample_coba_unbatchified
        else:
            decode_fn = self.sample_regular
        return decode_fn(input_ids=input_ids, 
                         logits_processor=logits_processor, 
                         stopping_criteria=stopping_criteria, 
                         max_length=max_length, 
                         pad_token_id=pad_token_id, 
                         eos_token_id=eos_token_id, 
                         output_attentions=output_attentions, 
                         output_hidden_states=output_hidden_states, 
                         output_scores=output_scores,
                         return_dict_in_generate=return_dict_in_generate,
                         synced_gpus=synced_gpus, 
                         streamer=streamer, 
                         uncond_dict=uncond_dict,
                         **model_kwargs)
    
    def sample_regular(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        logits_warper,
        max_length,
        pad_token_id,
        eos_token_id,
        output_attentions,
        output_hidden_states,
        output_scores,
        return_dict_in_generate,
        synced_gpus,
        streamer,
        uncond_dict,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        
    def sample_coba_unbatchified(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        logits_warper,
        max_length,
        pad_token_id,
        eos_token_id,
        output_attentions,
        output_hidden_states,
        output_scores,
        return_dict_in_generate,
        synced_gpus,
        streamer,
        uncond_dict,
        **model_kwargs,
    ):
        self._coba_steps = None
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        
        STATE_BACKTRACK = 1
        STATE_FORWARD = 2
        FILL_TOKEN = -float("Inf")

        probs_unmasked_adjust = []
        probs_masked_adjust = []
        scores_masked_adjust = []
        top_level_raw_logits = None
        top_level_warped_logits = None

        if self.adjust_by_unconditional:
            input_ids_uncond = uncond_dict['input_ids_uncond']
            model_kwargs_uncond = uncond_dict['model_kwargs_uncond']

        assert input_ids.shape[0] == 1, 'does not support batch decoding yet'

        STATE = STATE_FORWARD # start with forward mode
        COBA_MAX_STEPS = self.coba_max_steps
        coba_steps = 0
        outputs = None

        # auto-regressive generation
        while True:
            if coba_steps >= COBA_MAX_STEPS:
                this_peer_finished = True
                if not synced_gpus:
                    break
                continue

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            ################### BACKTRACK STATE ######################
            if STATE == STATE_BACKTRACK:
                coba_steps += 1
                next_token_prob_pre_warp = probs_masked_adjust[-1]
                next_token_scores = scores_masked_adjust[-1]
                # 3-leveling thingy for the top level
                if input_ids.shape[-1] == 1 and torch.allclose(next_token_scores, torch.tensor(-float('Inf')).cuda()):
                    if (top_level_warped_logits is not None) and (not torch.allclose(top_level_warped_logits, torch.tensor(FILL_TOKEN).cuda())):
                        scores_masked_adjust[-1] = top_level_warped_logits
                        top_level_warped_logits = None
                    else:
                        scores_masked_adjust[-1] = top_level_raw_logits
                        top_level_warped_logits = None
                        top_level_raw_logits = None
                    next_token_scores = scores_masked_adjust[-1]

                state_transition = True
                if input_ids.shape[-1] > 1:
                    prob_test = next_token_prob_pre_warp.max() >= self.coba_prob_thresh
                    if not prob_test:
                        state_transition = False
                    elif torch.allclose(next_token_scores, torch.tensor(FILL_TOKEN).cuda()):
                        state_transition = False
                    
                    if state_transition and self.coba_include_dist:
                        all_possible_next_tokens = torch.where(next_token_scores.flatten() > FILL_TOKEN)[0]
                        assert len(all_possible_next_tokens) < 6
                        for tok_idx in range(len(all_possible_next_tokens)):
                            tok = all_possible_next_tokens[tok_idx:tok_idx+1]
                            tok_dist = self.compute_tok_dist_to_context(tok.unsqueeze(1)).item()
                            if tok_dist > self.coba_dist_thresh:
                                scores_masked_adjust[-1][0,tok.item()] = FILL_TOKEN
                                probs_masked_adjust[-1][0,tok.item()] = FILL_TOKEN

                        next_token_scores = scores_masked_adjust[-1]
                        if torch.allclose(next_token_scores, torch.tensor(FILL_TOKEN).cuda()):
                            state_transition = False
                
                if not state_transition:
                    probs_masked_adjust = probs_masked_adjust[:-1]
                    probs_unmaksed_adjust = probs_unmasked_adjust[:-1]
                    scores_masked_adjust = scores_masked_adjust[:-1]

                    cur_last_token = input_ids[0,-1]
                    input_ids = input_ids[:,:-1]

                    if self.adjust_by_unconditional:
                        input_ids_uncond = input_ids_uncond[:,:-1]
                    
                    probs_masked_adjust[-1][0,cur_last_token.item()] = FILL_TOKEN
                    scores_masked_adjust[-1][0,cur_last_token.item()] = FILL_TOKEN
                    if input_ids.shape[-1] == 1:
                        if top_level_warped_logits is not None:
                            top_level_warped_logits[0,cur_last_token.item()] = FILL_TOKEN
                        if top_level_raw_logits is not None:
                            top_level_raw_logits[0,cur_last_token.item()] = FILL_TOKEN

                    if return_dict_in_generate:
                        if output_scores:
                            scores = scores[:-1]
                        if output_attentions:
                            decoder_attentions = decoder_attentions[:-1]
                            if self.config.is_encoder_decoder:
                                cross_attentions = cross_attentions[:-1]
                    continue
                else:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    # finished sentences should have their next token be a padding token
                    if eos_token_id is not None:
                        if pad_token_id is None:
                            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                    input_ids = torch.cat([input_ids, next_tokens[:,None]], dim=-1)
                    if self.adjust_by_unconditional:
                        input_ids_uncond = torch.cat([input_ids_uncond, next_tokens[:,None]], dim=-1)
                    STATE = STATE_FORWARD
                    
                    # if eos_token was found in one sentence, set sentence to finished
                    if eos_token_id_tensor is not None:
                        unfinished_sequences = unfinished_sequences.mul(
                            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                        )

                        # stop when each sentence is finished
                        if unfinished_sequences.max() == 0:
                            this_peer_finished = True

                    # stop if we exceed the maximum length
                    if stopping_criteria(input_ids, scores):
                        this_peer_finished = True

                    if this_peer_finished and not synced_gpus:
                        break
                    
                    continue
            
            ################### FORWARD STATE
            else:
                coba_steps += 1
                # prepare model inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                if self.adjust_by_unconditional:
                    model_inputs_uncond = self.model.prepare_inputs_for_generation(input_ids_uncond, **model_kwargs_uncond)
                
                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                if self.adjust_by_unconditional:
                    outputs_uncond = self.model(
                        **model_inputs_uncond, 
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states
                    )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                next_token_logits = outputs.logits[:, -1, :]
                # pre-process distribution
                next_token_logits_lp = logits_processor(input_ids, next_token_logits)
                if self.adjust_by_unconditional:
                    next_token_logits_uncond = outputs_uncond.logits[:,-1,:]
                    next_token_logits_lp_uncond = logits_processor(input_ids_uncond, next_token_logits_uncond)
                    next_token_logits_lp = logits_processor(input_ids, 
                            (1+self.adjust_by_unconditional_scale)*next_token_logits-self.adjust_by_unconditional_scale*next_token_logits_uncond
                            )
                
                if input_ids.shape[-1] == 1:
                    top_level_raw_logits = next_token_logits_lp.clone().detach()
                #############
                state_transition = False
                next_token_probs_pre_warper = F.softmax(next_token_logits_lp, dim=-1)
                next_token_probs_pre_warper_masked = next_token_probs_pre_warper.clone().detach()
                prob_test = next_token_probs_pre_warper.max() >= self.coba_prob_thresh
                if input_ids.shape[-1] > 1 and (not prob_test):
                    state_transition = True
                else:
                #############
                    next_token_scores = logits_warper(input_ids, next_token_logits_lp)
                    if input_ids.shape[-1] == 1:
                        top_level_warped_logits = next_token_scores.clone().detach()

                    next_token_scores[next_token_probs_pre_warper < self.coba_prob_thresh] = FILL_TOKEN
                    if torch.allclose(next_token_scores, torch.tensor(FILL_TOKEN).cuda()):
                        if input_ids.shape[-1] == 1:
                            next_token_scores = top_level_warped_logits.clone().detach()
                        else:
                            state_transition = True
                    
                    if input_ids.shape[-1] > 1 and (not state_transition) and self.coba_include_dist:
                        all_possible_next_tokens = torch.where(next_token_scores.flatten() > FILL_TOKEN)[0]
                        assert len(all_possible_next_tokens) < 6
                        for tok_idx in range(len(all_possible_next_tokens)):
                            tok = all_possible_next_tokens[tok_idx:tok_idx+1]
                            tok_dist = self.compute_tok_dist_to_context(tok.unsqueeze(1)).item()
                            if tok_dist > self.coba_dist_thresh:
                                next_token_scores[0,tok.item()] = FILL_TOKEN
                                next_token_probs_pre_warper_masked[0,tok.item()] = FILL_TOKEN

                        if torch.allclose(next_token_scores, torch.tensor(FILL_TOKEN).cuda()):
                            state_transition = True
       
                if state_transition:
                    cur_last_token = input_ids[0,-1]
                    probs_masked_adjust[-1][:,cur_last_token] = FILL_TOKEN
                    scores_masked_adjust[-1][:,cur_last_token] = FILL_TOKEN
                    input_ids = input_ids[:,:-1]
                    if input_ids.shape[-1] == 1: # if you only have one token left, then the token you removed is a top level token
                        if top_level_warped_logits is not None:
                            top_level_warped_logits[:,cur_last_token.item()] = FILL_TOKEN
                        if top_level_raw_logits is not None:
                            top_level_raw_logits[:,cur_last_token.item()] = FILL_TOKEN
                    
                    if self.adjust_by_unconditional:
                        input_ids_uncond = input_ids_uncond[:,:-1]
                    
                    STATE = STATE_BACKTRACK
                    continue

                # no state transition
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                probs_masked_adjust.append(next_token_probs_pre_warper_masked)
                probs_unmasked_adjust.append(next_token_probs_pre_warper)
                scores_masked_adjust.append(next_token_scores)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if self.adjust_by_unconditional:
                    input_ids_uncond = torch.cat([input_ids_uncond, next_tokens[:,None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if self.adjust_by_unconditional:
                    model_kwargs_uncond = self._update_model_kwargs_for_generation(
                        outputs_uncond, model_kwargs_uncond, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                    # stop when each sentence is finished
                    if unfinished_sequences.max() == 0:
                        this_peer_finished = True

                # stop if we exceed the maximum length
                if stopping_criteria(input_ids, scores):
                    this_peer_finished = True

                if this_peer_finished and not synced_gpus:
                    break
        
        self._coba_steps = [coba_steps]

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        **kwargs,
    ):

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        assert streamer is None; "streamer not supported yet"
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.model._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        model_kwargs_orig = copy.deepcopy(model_kwargs)
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        model_kwargs_orig.pop('attention_mask')
        if self.adjust_by_unconditional:
            inputs_tensor_uncond, model_input_name_uncond, model_kwargs_uncond = self.model._prepare_model_inputs(
                self.inputs_uncond.clone().detach(), generation_config.bos_token_id, model_kwargs_orig
            )

        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        if self.adjust_by_unconditional:
            model_kwargs_uncond['output_attentions'] = generation_config.output_attentions
            model_kwargs_uncond['output_hidden_states'] = generation_config.output_hidden_states


        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
            if self.adjust_by_unconditional:
                model_kwargs_uncond["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache
            if self.adjust_by_unconditional:
                model_kwargs_uncond["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if self.attention_mask_uncond is not None:
            model_kwargs_uncond['attention_mask'] = self.attention_mask_uncond

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )
        if self.adjust_by_unconditional and model_kwargs_uncond.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs_uncond["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor_uncond, generation_config.pad_token_id, generation_config.eos_token_id
            )
            self.attention_mask_uncond = model_kwargs_uncond["attention_mask"]

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
            if self.adjust_by_unconditional:
                model_kwargs_uncond = self.model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor_uncond, model_kwargs_uncond, model_input_name_uncond
                )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
            if self.adjust_by_unconditional:
                input_ids_uncond, model_kwargs_uncond = self.model._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size, 
                    model_input_name=model_input_name_uncond,
                    model_kwargs=model_kwargs_uncond, 
                    decoder_start_token_id=generation_config.decoder_start_token_id, 
                    bos_token_id=generation_config.bos_token_id, 
                    device=inputs_tensor_uncond.device,
                )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
            if self.adjust_by_unconditional:
                input_ids_uncond = inputs_tensor_uncond if model_input_name_uncond == 'input_ids' else model_kwargs_uncond.pop('input_ids')


        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
            (generation_config.num_beams == 1)
            and generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            if self.adjust_by_unconditional:
                uncond_dict = {'input_ids_uncond': input_ids_uncond, 
                               'model_kwargs_uncond': model_kwargs_uncond}
            else:
                uncond_dict = None
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                uncond_dict=uncond_dict,
                **model_kwargs,
            )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            if self.adjust_by_unconditional:
                assert generation_config.num_return_sequences is None or generation_config.num_return_sequences == 1, 'num seq error'
                input_ids_uncond, model_kwargs_uncond = self._expand_inputs_for_generation(
                    input_ids=input_ids_uncond, 
                    expand_size=generation_config.num_return_sequences, 
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    **model_kwargs_uncond
                )
                uncond_dict = {'input_ids_uncond': input_ids_uncond, 
                               'model_kwargs_uncond': model_kwargs_uncond}
            else:
                uncond_dict = None

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                uncond_dict=uncond_dict,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if generation_config.diversity_penalty == 0.0:
                raise ValueError(
                    "`diversity_penalty` should be greater than `0.0`, otherwise your beam groups will be identical."
                )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
