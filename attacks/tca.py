import torch
from torch.nn import CosineSimilarity


"""
Temporal Characteristics-based Adversarial Attack (TCA) aims to craft adversarial examples
that maximize the model's loss while preserving high cosine similarity with the original input.
This ensures the perturbations remain stealthy by maintaining temporal characteristics and
the overall direction of input sequences.

TCA can be seen as an extension of the Basic Iterative Method (BIM), where small gradient-based
perturbations are applied iteratively. However, at each step, the method selects updates that
maximize the cosine similarity between the adversarial and original inputs, thus preserving
the temporal structure and making the attack more stealthy.

for learn more about this attack :
https://www.sciencedirect.com/science/article/abs/pii/S0957417424028173
"""

def tca_attack(model, loss_fn, x, y, epsilon, alpha=None, num_iter=20):
    model.eval()
    x_adv = x.clone().detach()
    cos_sim = CosineSimilarity(dim=1)

    if alpha is None:
        alpha = epsilon / num_iter

    for _ in range(num_iter):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        output = model(x_adv)
        loss = loss_fn(output, y)
        loss.backward()

        eta = alpha * x_adv.grad.sign()
        x_tilde = x_adv + eta
        x_tilde = torch.clamp(x_tilde, x - epsilon, x + epsilon)

        sim_tilde = cos_sim(x.view(x.size(0), -1), x_tilde.detach().view(x.size(0), -1))
        sim_plus = cos_sim(x.view(x.size(0), -1), (x + epsilon).view(x.size(0), -1))
        sim_minus = cos_sim(x.view(x.size(0), -1), (x - epsilon).view(x.size(0), -1))

        x_adv_new = x_adv.clone().detach()
        for i in range(x.size(0)):
            if sim_tilde[i] > sim_plus[i] and sim_tilde[i] > sim_minus[i]:
                x_adv_new[i] = x_tilde[i]
            elif sim_plus[i] >= sim_minus[i]:
                x_adv_new[i] = (x + epsilon)[i]
            else:
                x_adv_new[i] = (x - epsilon)[i]

        x_adv = x_adv_new

    return x_adv.detach()