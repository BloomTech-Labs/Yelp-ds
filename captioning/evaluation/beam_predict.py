import numpy as np

from sequence_candidate import SequenceCandidate


def generate_predictions_beam(img_id, features, caption_model, reverse_tokenizer, width, num_neighbors,
                              top_n = 3, end_idx = 2, max_length = 15, ignore_idx = [4,61,345], alpha = .6):
    photo_features = features[img_id]
    accepted_sequences = []
    population = []
    start_sequence = SequenceCandidate.template_seq(ignore_idx = ignore_idx, alpha = alpha)
    population.append(start_sequence)
    for i in range(max_length - 1):
        tmp = []
        for cand_seq in population:
            pred = caption_model.predict([photo_features, cand_seq._seq.reshape(1,-1)], verbose=0)[0]
            pred_argsort = pred.argsort()
            for next_idx in pred_argsort[-num_neighbors:]:
                if (cand_seq.final_token(), next_idx) in cand_seq._bigrams:
                    accepted_sequences.append(cand_seq)
                    continue
                next_prob = pred[next_idx]
                new_candidate = cand_seq.add_token(next_idx,next_prob)
                if next_idx == end_idx:
                    accepted_sequences.append(new_candidate)
                else:
                    tmp.append(new_candidate)
        try:
            population = sorted(tmp)[-width:]
        except:
            population = tmp
            break

    accepted_sequences = sorted(accepted_sequences + population, reverse = True)
    num_accepted = 0
    values = []
    probs = []
    strings = []
    for acc_seq in accepted_sequences:
        seq_string = acc_seq.to_words(reverse_tokenizer,end_idx)
        if seq_string not in strings:
            strings.append(seq_string)
            probs.append(acc_seq.probsum())
            num_accepted += 1
            if num_accepted >= top_n:
                break
    output = list(zip(strings,probs))
    return output
