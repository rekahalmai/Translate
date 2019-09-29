from torch.autograd import Variable
from lib.beam import nopeak_mask, k_best_outputs
import torch.nn.functional as F
import math
from nltk.corpus import wordnet
import torch

def get_synonym(word, SRC, src_lang):
    if src_lang == 'fr':
        syns = wordnet.synsets(word, lang='fra')
        for s in syns:
            for l in s.lemmas():
                if SRC.vocab.stoi[l.name()] != 0:
                    return SRC.vocab.stoi[l.name()]
        return 0
    else:
        syns = wordnet.synsets(word)
        for s in syns:
            for l in s.lemmas():
                if SRC.vocab.stoi[l.name()] != 0:
                    return SRC.vocab.stoi[l.name()]

        return 0



def translate_text(src_text, params, model, SRC, TRG):
    # Model in evaluation mode
    model.eval()
    # word/ number mapping
    indexed = []
    # In case we translate more than one sentence, keep the sentence count
    num_finished_sentences = None

    # preprocess the text according to the source preprocess
    sentence = SRC.preprocess(src_text)

    for tok in sentence:
        # when the token is known
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC, params["src_lang"]))

    # Save the sentence as a torch censor and send it to the GPU
    sentence = Variable(torch.LongTensor([indexed]))
    sentence.cuda()

    # Start of sequence token
    init_tok = TRG.vocab.stoi['<sos>']
    # source mask when the text is padded. Send it to GPU
    src_mask = (sentence != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    src_mask.cuda()

    # Encoder output
    e_output = model.encoder(sentence.cuda(), src_mask.cuda())
    outputs = torch.LongTensor([[init_tok]]).cuda()

    # Target masking
    trg_mask = nopeak_mask(1, params)

    # Output of decoder
    out = model.out(model.decoder(outputs.cuda(), e_output.cuda(), src_mask.cuda(), trg_mask.cuda())).cuda()
    out = F.softmax(out, dim=-1)

    # Top k outputs probabilities and indexes
    probs, ix = out[:, -1].data.topk(params["k"])
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    outputs = torch.zeros(params["k"], params["max_length"]).long().cuda()

    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(params["k"], e_output.size(-2), e_output.size(-1))

    e_outputs[:, :] = e_output[0]
    e_outputs.cuda()

    outputs, e_outputs, log_scores = outputs.cuda(), e_outputs.cuda(), log_scores.cuda()

    # End of sentence token
    eos_tok = TRG.vocab.stoi['<eos>']

    ind = None
    for i in range(2, params["max_length"]):
        trg_mask = nopeak_mask(i, params)

        out = model.out(model.decoder(outputs[:, :i].cuda(), e_outputs.cuda(), src_mask.cuda(), trg_mask.cuda()))
        out = F.softmax(out.cuda(), dim=-1)

        outputs, log_scores = k_best_outputs(outputs.cuda(), out.cuda(), log_scores.cuda(), i, params["k"])
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()

        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol
            num_finished_sentences = len([s for s in sentence_lengths if s > 0])
            # print(num_finished_sentences)

        if num_finished_sentences == params["k"]:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores.cuda() * div.cuda(), 1)
            ind = ind.data[0]
            break
            
    try: 
        length = (outputs[0] == eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    except (IndexError, RuntimeError): 
        length = (outputs[0] == eos_tok).nonzero()
        return ''
    
