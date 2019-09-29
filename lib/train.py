import os
import time
import torch
import torch.nn.functional as F
from lib.batch import create_masks

def train_model(model, params, model_name):

    print("training model...")
    model.train()
    start = time.time()
    if params["checkpoint"] > 0:
        cptime = time.time()

    avg_loss = 0

    for epoch in range(params["epoch"]):

        total_loss = 0
        print("   %dm: epoch %d [%s]  %d%%  loss = %s" % \
              ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

        if params["checkpoint"] > 0: 
            torch.save(model, "model/" + model_name)
           
        for i, batch in enumerate(params["train"]):

            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, params)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            params["optimizer"].zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=params["trg_pad"])
            loss.backward()
            params["optimizer"].step()

            if params["SGDR"] == True:
                params["sched"].step()

            total_loss += loss.item()

            if (i + 1) % params["printevery"] == 0:
                p = int(100 * (i + 1) / params["train_len"])
                avg_loss = total_loss / params["printevery"]
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))),
                       p, avg_loss), end='\r')
                total_loss = 0

            if params["checkpoint"] > 0 and ((time.time() - cptime) // 60) // params["checkpoint"] >= 1:
                torch.save(model, 'model/' + model_name)
                cptime = time.time()

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))
        