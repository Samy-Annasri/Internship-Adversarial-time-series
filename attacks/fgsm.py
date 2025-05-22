"""
Fast Gradient Sign Method (FGSM) generates adversarial examples by adding
a small perturbation to the input based on the gradient of the loss.

The perturbation is computed in the direction that increases the model's
loss the most.

But In the contexte of time-series data, the perturbations must be
handled more carefully to remain realistic.

This is easy to add a lot of noise but that is going to be really easy to
see the attack

Formula:
    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
"""
def fgsm_attack(model, loss_fn, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    model.eval()

    output = model(x_adv)
    loss = loss_fn(output, y)
    loss.backward()

    # Compute perturbation using the sign of the gradient
    perturbation = epsilon * x_adv.grad.sign()
    # Apply perturbation to the input
    x_adv = x_adv + perturbation
    return x_adv.detach()