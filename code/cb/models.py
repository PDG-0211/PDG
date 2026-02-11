# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class BaseCodeBert(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
        self.config = config_class.from_pretrained(path)
        self.tokenizer = tokenizer_class.from_pretrained(path)
        self.encoder = model_class.from_pretrained(path,config=self.config)  
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, 
                                                   nhead=self.config.num_attention_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model=Seq2Seq(tokenizer=self.tokenizer, encoder=self.encoder, decoder=self.decoder, config=self.config,
                           beam_size=args.beam_size, max_length=args.max_output_len, sos_id=self.tokenizer.bos_token_id, eos_id=self.tokenizer.eos_token_id, args=args)
    def forward(self, source_ids, source_mask, target_ids, target_mask):
        loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
        return loss
    
    def generate(self, source_ids, source_mask):
        p = []
        preds = self.model(source_ids=source_ids,source_mask=source_mask) 
        for pred in preds:
            t=pred[0].cpu().numpy()
            t=list(t)
            if 0 in t:
                t=t[:t.index(0)]
            text =self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
            p.append(text)
        return ' '.join(p)
    
class MultiEncoderCodeBert(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
        self.config = config_class.from_pretrained(path)
        self.tokenizer = tokenizer_class.from_pretrained(path)

        self.encoder1 = model_class.from_pretrained(path,config=self.config)    
        self.encoder2 = model_class.from_pretrained(path,config=self.config)    
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, 
                                                   nhead=self.config.num_attention_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model=Seq2Seq2(tokenizer=self.tokenizer, encoder1=self.encoder1, encoder2=self.encoder2, decoder=self.decoder, config=self.config,
                           beam_size=args.beam_size, max_length=args.max_output_len, sos_id=self.tokenizer.bos_token_id, eos_id=self.tokenizer.eos_token_id, args=args)
    def forward(self, source_ids, source_mask,
                 source_ids2, source_mask2,
                 target_ids, target_mask):
        loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, 
                                source_ids2=source_ids2, source_mask2=source_mask2,
                                target_ids=target_ids, target_mask=target_mask)
        return loss
    
    def generate(self, source_ids, source_mask,
                 source_ids2, source_mask2):
        p = []
        preds = self.model(source_ids=source_ids,source_mask=source_mask,
                           source_ids2=source_ids2,source_mask2=source_mask2) 
        for pred in preds:
            t=pred[0].cpu().numpy()
            t=list(t)
            if 0 in t:
                t=t[:t.index(0)]
            text =self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
            p.append(text)
        return ' '.join(p)
  
class Seq2Seq2(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, tokenizer, encoder1, encoder2, decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None, args=None):
        super(Seq2Seq2, self).__init__()
        self.tokenizer = tokenizer
        self.encoder1 = encoder1   
        self.encoder2  = encoder2    

        self.tgt_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.args=args
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder1.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None, source_ids2=None, source_mask2=None, target_ids=None,target_mask=None):   
        outputs1 = self.encoder1(source_ids, attention_mask=source_mask)
        outputs2 = self.encoder2(source_ids2, attention_mask=source_mask2)

        encoder_output1 = outputs1[0]
        encoder_output2 = outputs2[0]

        encoder_output = torch.cat((encoder_output1, encoder_output2), dim=1)
        encoder_output = encoder_output.permute([1,0,2]).contiguous()

        encoder_mask = torch.cat((source_mask, source_mask2), dim=1)

        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])

            tgt_embeddings = self.encoder1.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-encoder_mask).bool())
            
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero = torch.LongTensor(1).fill_(0).to(source_ids.device) 
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask = encoder_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id, self.args)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])

                    tgt_embeddings = self.encoder1.embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds   


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, tokenizer, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None, args=None):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.args=args
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero = torch.LongTensor(1).fill_(0).to(source_ids.device) 
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id, self.args)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds   
        
class Beam(object):
    def __init__(self, size, sos, eos, args):
        self.size = size
        self.device = torch.device("cuda", args.device)
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(self.device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(0).to(self.device)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        batch = self.nextYs[-1].view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def advance(self, wordLk):
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
