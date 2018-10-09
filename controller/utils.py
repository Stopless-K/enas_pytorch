from torch import  nn
import torch.nn.functional as F

def process_logits(logits, temperature, tanh_constant):

    if temperature:
        logits = logits/ temperature
    if tanh_constant:
        logits = tanh_constant* nn.Tanh(logits)

    return logits


def sample(logits):

    probs = F.softmax(logits, -1)

    action = probs.multinomial(num_samples=1).squeeze()
    action = action.view(1)

    log_probs = F.log_softmax(logits, -1)

    pi_log_probs = log_probs[:, action]


    entropy = -(probs* log_probs).sum(1,True)
    # print(entropy.size())
    return action, pi_log_probs, entropy


