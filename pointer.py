import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable

import data
import model

from splitcross import SplitCrossEntropyLoss
from utils import batchify, get_batch, repackage_hidden


def one_hot(idx, size, cuda=True):
    a = np.zeros((1, size), np.float32)
    a[0][idx] = 1
    # v = Variable(torch.from_numpy(a))
    v = torch.from_numpy(a)
    if cuda:
        v = v.cuda()
    return v


def evaluate(model, data_source, ntokens, args, batch_size=10):
    window = args.window

    # Turn on evaluation mode which disables dropout.
    if args.model == 'QRNN':
        model.reset()

    print(model)
    model.eval()
    total_loss = 0
    # ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    next_word_history = None
    pointer_history = None
    for i in range(0, data_source.size(0) - 1, args.bptt):
        if i > 0:
            print(i, len(data_source), math.exp(total_loss / i))
        data, targets = get_batch(data_source, i, evaluation=True, args=args)
        output, hidden, rnn_outs, _ = model(data, hidden, return_h=True)
        rnn_out = rnn_outs[-1].squeeze()
        output_flat = output.view(-1, ntokens)
        ###
        # Fill pointer history
        start_idx = (len(next_word_history)
                     if next_word_history is not None else 0)
        next_word_history = torch.cat([one_hot(t.item(), ntokens)
                                       for t in targets]) if next_word_history is None else torch.cat(
            [next_word_history, torch.cat([one_hot(t.item(), ntokens) for t in targets])])
        # print(next_word_history)
        # pointer_history = (Variable(rnn_out.data)
        #                    if pointer_history is None
        #                    else torch.cat([pointer_history,
        #                                    Variable(rnn_out.data)], dim=0))
        pointer_history = (rnn_out.detach()
                           if pointer_history is None
                           else torch.cat([pointer_history,
                                           rnn_out.detach()], dim=0))
        # print(pointer_history)
        ###
        # Built-in cross entropy
        # total_loss += len(data) * criterion(output_flat, targets).data[0]
        ###
        # Manual cross entropy
        # softmax_output_flat = torch.nn.functional.softmax(output_flat)
        # soft = torch.gather(softmax_output_flat, dim=1, index=targets.view(-1, 1))
        # entropy = -torch.log(soft)
        # total_loss += len(data) * entropy.mean().data[0]
        ###
        # Pointer manual cross entropy
        loss = 0
        softmax_output_flat = torch.nn.functional.softmax(output_flat, dim=-1)
        for idx, vocab_loss in enumerate(softmax_output_flat):
            p = vocab_loss
            if start_idx + idx > window:
                valid_next_word = next_word_history[
                    start_idx + idx - window:start_idx + idx]
                valid_pointer_history = pointer_history[
                    start_idx + idx - window:start_idx + idx]
                logits = torch.mv(valid_pointer_history, rnn_out[idx])
                theta = args.theta
                ptr_attn = (torch.nn.functional.softmax(theta * logits, dim=-1)
                            .view(-1, 1))
                ptr_dist = (ptr_attn.expand_as(valid_next_word)
                            * valid_next_word).sum(0).squeeze()
                lambdah = args.lambdasm
                p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss
            ###
            target_loss = p[targets[idx].data]
            loss += (-torch.log(target_loss)).item()
        total_loss += loss / batch_size
        ###
        hidden = repackage_hidden(hidden)
        next_word_history = next_word_history[-window:]
        pointer_history = pointer_history[-window:]
    return total_loss / len(data_source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN)')
    parser.add_argument('--save', type=str, default='best.pt',
                        help='model to use the pointer over')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--bptt', type=int, default=3785,
                        help='sequence length')
    parser.add_argument('--window', type=int, default=5000,
                        help='pointer window length')
    parser.add_argument('--theta', type=float, default=0.6625523432485668,
                        help='mix between uniform distribution and pointer softmax distribution over previous words')
    parser.add_argument('--lambdasm', type=float, default=0.12785920428335693,
                        help='linear mix between only pointer (1) and only vocab (0) distribution')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    args = parser.parse_args()

    ###########################################################################
    # Load data
    ###########################################################################

    corpus = data.Corpus(args.data)

    eval_batch_size = 1
    test_batch_size = 1
    # train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, test_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ###########################################################################
    # Build the model
    ###########################################################################

    ntokens = len(corpus.dictionary)

    # criterion = nn.CrossEntropyLoss()
    # master branch has a but here, see this:
    # https://github.com/salesforce/awd-lstm-lm/issues/28
    # it should not be using CrossEntropyLoss()
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits,
                                      verbose=False)

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        if not args.cuda:
            model = torch.load(f, map_location=lambda storage, loc: storage)
        else:
            model = torch.load(f)
    print(model)

    # Run on val data.
    val_loss = evaluate(model, val_data, ntokens, args, test_batch_size)
    print('=' * 89)
    print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    print('=' * 89, flush=True)

    # Run on test data.
    test_loss = evaluate(model, test_data, ntokens, args, test_batch_size)
    print('=' * 89)
    print('| End of pointer | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89, flush=True)
