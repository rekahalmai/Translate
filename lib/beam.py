import torch
from lib.batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, params):
    
    init_tok = TRG.vocab.stoi['<sos>']
    
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    if params["device"] == torch.device('cuda'): 
        src_mask.cuda()
        
    e_output = model.encoder(src, src_mask)
    outputs = torch.LongTensor([[init_tok]])
    if params["device"] == torch.device('cuda'): 
        outputs.cuda()
        e_output.cuda()
        
    trg_mask = nopeak_mask(1, params)
    if params["device"] == torch.device('cuda'): 
        trg_mask.cuda()
    
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask)).cuda()
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(params["k"])
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(params["k"], params["max_length"]).long()

    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(params["k"], e_output.size(-2),e_output.size(-1))

    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    probs, ix = probs.cuda(), ix.cuda()
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1).cuda() + log_scores.cuda().transpose(0,1)
    log_probs = log_probs.cuda()
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i].cuda()
    outputs[:, i] = ix[row, col].cuda()

    log_scores = k_probs.unsqueeze(0)
    log_scores = log_scores.cuda()
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, params):
    

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, params)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, params["max_length"]):
    
        trg_mask = nopeak_mask(i, params)

        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, params["k"])
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == params["k"]:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
