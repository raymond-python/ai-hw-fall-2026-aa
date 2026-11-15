"""White-box, untargeted L-infinity adversarial attacks.

All attacks take raw images x in [0, 1] with true labels y and return x_adv
such that  max |x_adv - x| <= eps  and  0 <= x_adv <= 1.
Each step pushes the image in the direction that INCREASES the loss of the
true class, i.e. away from the correct answer.

    FGSM     one step of size eps along sign(grad)            (Goodfellow 2015)
    I-FGSM   many small steps, clipped to the eps-ball        (Kurakin 2017)
    PGD      I-FGSM + random start + bigger step              (Madry 2018)
    MI-FGSM  I-FGSM with a momentum term on the gradient      (Dong 2018)
"""
import torch
import torch.nn.functional as F


def _grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (g,) = torch.autograd.grad(loss, x)
    return g.detach()


def _project(x_adv, x, eps):
    """Clip back into the eps-ball around x and into the valid pixel range."""
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.clamp(0.0, 1.0).detach()


def fgsm(model, x, y, eps, **_):
    return _project(x + eps * _grad(model, x, y).sign(), x, eps)


def ifgsm(model, x, y, eps, steps=10, alpha=None, **_):
    alpha = eps / steps if alpha is None else alpha
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def pgd(model, x, y, eps, steps=10, alpha=None, **_):
    alpha = 2.5 * eps / steps if alpha is None else alpha
    x_adv = _project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def mifgsm(model, x, y, eps, steps=10, alpha=None, mu=1.0, **_):
    alpha = eps / steps if alpha is None else alpha
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)
    for _ in range(steps):
        grad = _grad(model, x_adv, y)
        # normalise each image's gradient by its L1 norm, then accumulate
        l1 = grad.abs().flatten(1).sum(1).clamp_min(1e-12).view(-1, 1, 1, 1)
        g = mu * g + grad / l1
        x_adv = _project(x_adv + alpha * g.sign(), x, eps)
    return x_adv


ATTACKS = {"FGSM": fgsm, "I-FGSM": ifgsm, "PGD": pgd, "MI-FGSM": mifgsm}
