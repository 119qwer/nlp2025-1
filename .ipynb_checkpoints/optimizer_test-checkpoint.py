import torch
import numpy as np
from optimizer import AdamW

SEED = 0


def test_optimizer(opt_class) -> torch.Tensor:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        correct_bias=True,
    )
    for i in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()

if __name__ == '__main__':
    ref = torch.tensor(np.load("optimizer_test.npy")) #이 파일이 optimizer의 정답 가중치임
    actual = test_optimizer(AdamW)
    print(ref)
    print(actual)
    assert torch.allclose(ref, actual, atol=1e-6, rtol=1e-4)
    print("Optimizer test passed!")
