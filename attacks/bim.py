import torch

'''
The Basic Iterative Method (BIM) is an extension of FGSM that applies the attack iteratively,
performing multiple small perturbations instead of a single step.
It also includes a clipping mechanism to ensure the total perturbation stays within the epsilon limit.
'''
def bim_attack(model, loss_fn, x, y, epsilon, alpha, num_iter):
    model.eval()
    x_adv = x.clone().detach()
    eta = torch.zeros_like(x_adv)

    for _ in range(num_iter):
        x_adv_ = torch.clamp(x + eta, 0, 1).detach().clone().requires_grad_(True)
        output = model(x_adv_)
        loss = loss_fn(output, y)

        model.zero_grad()
        loss.backward()

        grad = x_adv_.grad.data
        eta = eta + alpha * grad.sign()
        eta = torch.clamp(eta, -epsilon, epsilon)

    x_adv_final = torch.clamp(x + eta, 0, 1)
    return x_adv_final.detach()