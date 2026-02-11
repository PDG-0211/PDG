import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Stack, T5EncoderModel, T5Config, T5ForConditionalGeneration, BaseModelOutputWithPastAndCrossAttentions
import copy
from torch.nn import CrossEntropyLoss
import time
import logging
import math
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# codeT5，encoder-decoder
class BaseT5model(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    def generate(self, input_ids, attention_mask):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
# codeT5-encoder + Transformer-decoder
class T5OnlyEncoderModel(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        self.config = T5Config.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.encoder = T5EncoderModel.from_pretrained(path)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads)
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

# encoder-concat
class MultiEncoderT5Model(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        self.args = args
        self.config = T5Config.from_pretrained(path)
        # 两个编码器都用CodeT5
        self.encoder1=T5EncoderModel.from_pretrained(path)    
        self.encoder2=T5EncoderModel.from_pretrained(path)    
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        # 模型
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels, return_encoder=False):
        encoder_output1 = self.encoder1(input_ids=input_ids1, 
                                        attention_mask=attention_mask1)
        encoder_output2 = self.encoder2(input_ids=input_ids2, 
                                        attention_mask=attention_mask2)
        last_hidden_states1 = encoder_output1.last_hidden_state
        last_hidden_states2 = encoder_output2.last_hidden_state

        hidden_states = torch.cat([last_hidden_states1, last_hidden_states2], dim=1)

        hidden_states = self.layer_norm(hidden_states)

        
        encoder_attention_mask = torch.cat([attention_mask1, attention_mask2], dim=1)
        

        if return_encoder:

            encoder_output = copy.deepcopy(encoder_output1)
            encoder_output.last_hidden_state = hidden_states
            encoder_output.attention_mask = encoder_attention_mask
            return encoder_output, encoder_attention_mask
        
        return self.model(encoder_outputs=(hidden_states,), attention_mask=encoder_attention_mask, labels=labels)

    def generate(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        encoder_output, encoder_attention_mask = self.forward(input_ids1=input_ids1, attention_mask1=attention_mask1,
                                                             input_ids2=input_ids2, attention_mask2=attention_mask2, 
                                                             labels=None, return_encoder=True)

        generated_ids = self.model.generate(encoder_outputs=encoder_output, attention_mask=encoder_attention_mask)
        return generated_ids

class MultiEncoderFusionT5Model(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        self.args = args
        self.config = T5Config.from_pretrained(path)

        self.encoder1=T5EncoderModel.from_pretrained(path)  
        self.encoder2=T5EncoderModel.from_pretrained(path)   

        self.encoder1_attention=MultiHeadedAttention(head_count=self.config.num_attention_heads, 
                                                     model_dim=self.config.hidden_size)
                                                     
        self.encoder2_attention=MultiHeadedAttention(head_count=self.config.num_attention_heads,
                                                        model_dim=self.config.hidden_size)
        
        
        self.dropout = nn.Dropout(self.config.dropout_rate)

        self.layer_norm = nn.LayerNorm(self.config.hidden_size)


        self.model = T5ForConditionalGeneration.from_pretrained(path)
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels, return_encoder=False):
        encoder_output1 = self.encoder1(input_ids=input_ids1, 
                                        attention_mask=attention_mask1)
        encoder_output2 = self.encoder2(input_ids=input_ids2, 
                                        attention_mask=attention_mask2)
        
        last_hidden_states1 = encoder_output1.last_hidden_state
        last_hidden_states2 = encoder_output2.last_hidden_state
        
        mask1 = attention_mask1.unsqueeze(1).bool()
        mask2 = attention_mask2.unsqueeze(1).bool()
        residual = last_hidden_states2
        selt_attention_output, _ = self.encoder2_attention(key=last_hidden_states2,
                                                           value=last_hidden_states2,
                                                           query=last_hidden_states2,
                                                           mask=mask2)
        cross_attention_input = self.dropout(selt_attention_output) + residual

        cross_residual = cross_attention_input
        cross_attention_input = self.layer_norm(cross_attention_input)
        cross_attention_output, _ = self.encoder1_attention(key=last_hidden_states1,
                                                            value=last_hidden_states1,
                                                            query=cross_attention_input,
                                                            mask=mask1)
        hidden_states = cross_attention_output + cross_residual

        if return_encoder:
            encoder_output = copy.deepcopy(encoder_output1)
            encoder_output.last_hidden_state = hidden_states
            return encoder_output

        return self.model(encoder_outputs=(hidden_states,),
                          labels=labels)
    
    def generate(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        encoder_output = self.forward(input_ids1=input_ids1, attention_mask1=attention_mask1,
                                                             input_ids2=input_ids2, attention_mask2=attention_mask2, 
                                                             labels=None, return_encoder=True)

        generated_ids = self.model.generate(encoder_outputs=encoder_output)
        return generated_ids

# deprecated
class MultiEmbeddingT5Model(nn.Module):
    def __init__(self, path, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.embedding = self.model.get_input_embeddings()
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels, return_encoder=False):
        embedding1 = self.embedding(input_ids1)
        embedding2 = self.embedding(input_ids2)
        embedding = torch.cat([embedding1, embedding2], dim=1)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=1)
        if return_encoder:
            return embedding, attention_mask
        else:
            return self.model(inputs_embeds=embedding, attention_mask=attention_mask, labels=labels)
    def generate(self, input_ids1, attention_mask1, input_ids2, attention_mask2):

        embedding, attention_mask = self.forward(input_ids1, attention_mask1, 
                                                 input_ids2, attention_mask2, 
                                                 labels=None, return_encoder=True)

        generated_ids = self.model.generate(inputs_embeds=embedding, attention_mask=attention_mask)
        return generated_ids


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
                                   self.encoder.shared)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.shared(target_ids).permute([1,0,2]).contiguous()
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
                    tgt_embeddings = self.encoder.shared(input_ids).permute([1,0,2]).contiguous()
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

def generate_relative_position_matrix(length, max_relative_position, use_negative_distance):
    """
    Generate the clipped relative position matrix.
    """
    range_vec = torch.arange(length)
    range_matrix = range_vec.unsqueeze(1).expand(-1, length).transpose(0,1)
    distance_matrix = range_matrix - range_matrix.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_matrix, min=-max_relative_position, max=max_relative_position)

    if use_negative_distance:
        final_matrix = distance_mat_clipped + max_relative_position
    else:
        final_matrix = torch.abs(distance_mat_clipped)

    return final_matrix

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from 'Attention is all you need'.
    consider relative position.
    """
    def __init__(self, head_count: int, model_dim:int, dropout: float=0.1,
                 max_relative_position=0, use_negative_distance=False) -> None:
        super().__init__()
        assert model_dim % head_count == 0, 'model dim must be divisible by head count'

        self.head_size = model_dim // head_count
        self.head_count = head_count
        self.model_dim = model_dim

        self.key_project = nn.Linear(model_dim, head_count * self.head_size)
        self.query_project = nn.Linear(model_dim, head_count * self.head_size)
        self.value_project = nn.Linear(model_dim, head_count * self.head_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(model_dim, model_dim)

        self.max_relative_position = max_relative_position
        self.use_negative_distance = use_negative_distance

        if self.max_relative_position > 0:
            relative_position_size = self.max_relative_position*2+1 if self.use_negative_distance is True else self.max_relative_position+1
            self.relative_position_embedding_key = nn.Embedding(relative_position_size, self.head_size)
            self.relative_position_embedding_value = nn.Embedding(relative_position_size, self.head_size)

    def forward(self, key, value, query, mask=None):
        """
        Compute multi-headed attention.
        key  [batch_size, seq_len, hidden_size]
        value[batch_size, seq_len, hidden_size]
        query[batch_size, seq_len, hidden_size]
        mask [batch_size, 1 or seq_len, seq_len] (pad position is false or zero)

        return 
            - output [batch_size, query_len, hidden_size]
            - attention_output_weights [batch_size, query_len, key_len]
        """
        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)
        value_len = value.size(1)

        # project query key value
        key = self.key_project(key)
        value = self.value_project(value)
        query = self.query_project(query)

        #reshape key, value, query 
        key = key.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)
        #[batch_size, head_count, key_len, head_size]
        value = value.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)
        query = query.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)

        # scale and calculate attention scores
        query = query / math.sqrt(self.head_size)
        scores = torch.matmul(query, key.transpose(2,3))
        # scores [batch_size, head_count, query_len, key_len]

        if self.max_relative_position > 0: 
            relative_position_matrix = generate_relative_position_matrix(key_len, self.max_relative_position, self.use_negative_distance)
            relative_position_matrix = relative_position_matrix.to(key.device)
            relative_key = self.relative_position_embedding_key(relative_position_matrix)
            # relative_key [key_len, key_len, head_size]
            relative_vaule = self.relative_position_embedding_value(relative_position_matrix)
            # relative_value [value_len, value_len, head_size]
            r_query = query.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, self.head_size)
            assert query_len == key_len, "For relative position."
            scores_relative = torch.matmul(r_query, relative_key.transpose(1,2)).reshape(query_len, batch_size, self.head_count, key_len)
            scores_relative = scores_relative.permute(1, 2, 0, 3)
            scores = scores + scores_relative

        # apply mask Note: add a dimension to mask -> [batch_size, 1, 1 or len , key_len]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        # apply attention dropout
        attention_weights = self.softmax(scores) # attention_weights [batch_size, head_count, query_len, key_len]
        attention_probs = self.dropout(attention_weights)

        # get context vector
        context = torch.matmul(attention_probs, value) # context [batch_size, head_count, query_len, head_size]
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
        # context [batch_size, query_len, hidden_size]

        if self.max_relative_position > 0:
            r_attention_probs = attention_probs.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, key_len)
            context_relative = torch.matmul(r_attention_probs, relative_vaule) # context_relative [query_len, batch_size*self.head_count, head_size]
            context_relative = context_relative.reshape(query_len, batch_size, self.head_count, self.head_size).permute(1, 2, 0, 3)
            context_relative = context_relative.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
            # context_relative [batch_size, query_len, hidden_size]
            context = context + context_relative

        output = self.output_layer(context)

        attention_output_weights = attention_weights.view(batch_size, self.head_count, query_len, key_len).sum(dim=1) / self.head_count

        return output, attention_output_weights

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