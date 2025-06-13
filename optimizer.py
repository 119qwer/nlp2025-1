from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

#optimizer는 학습을 통해 모델의 파라미터(가중치)를 변화시키는 부분이다.
#adanW는 Adam + Weight Decay 알고리즘을 적용하는 것이다.
class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter], #학습할 모델의 파라미터, 일반적으로 model.parameters()를 전달
            lr: float = 1e-3, #학습률, 파라미터를 얼마나 크게 업데이트 할지 결정함, 보통 GPT-2에서는 5e-5 ~ 1e-4 정도로 작게 사용4
            eps: float = 1e-6, #분모가 0이 되는 것을 방지하는 작은 수
#Adam의 모멘텀 계수, 첫 번째 값(0.9): 과거 gradient의 평균(m)을 얼마나 반영할지, 두 번째 값(0.999): 과거 제곱 gradient(v)의 평균을 얼마나 반영할지
            betas: Tuple[float, float] = (0.9, 0.999), 
        #너무 큰 가중치는 더 크게 줄이는 방식으로 가중치들이 0에 수렴하게 만듬
            weight_decay: float = 0.0, #과적합을 방지하고 모델의 가중치가 너무 커지지 않도록 억제할 L2 정규화 항
            correct_bias: bool = True,#Adam은 초기 업데이트에서 bias가 생기는데, 이를 보정할지를 설정
    ):
        #초기화가 잘못되는 경우 에러 리리
        if lr < 0.0: #학습률이 0이하가 되면 에러
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0: #과거 그래디언트 평균이 0~1사이가 아니면 에러
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0: #과거 그래디언트의 제곱 평균이 0~1 사이가 이니면 러러
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps: #eps가 0이하면 러러
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        #옵티마이저 파라미터를 딕셔너리로 정의
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        #AdamW 클래스가 상속한 Optimizer 클래스의 생성자(__init__)를 명시적으로 호출하는 코드
        super().__init__(params, defaults)

    #모델 파라미터 업데이트 함수, 아무리 gradient를 계산하고 loss를 구해도, step()을 호출하지 않으면 모델은 학습이 안됨
    def step(self, closure: Callable = None):
        #loss를 재계산할 경우를 대비해서, loss라는 변수를 초기화
        loss = None
        #closure 인자가 넘어왔다면(사용자가 loss를 다시 계산하고 싶다고 설정) closure을 다시 실행해서 loss 재계산 
        #2차 최적화 알고리즘 (예: LBFGS)에 쓰이지만 AdamW은 거의 사용 안함@@(나중에 개선에 사용될수도?)
        if closure is not None:
            loss = closure()

        #여러 파라미터 그룹이 있을 수 있고 이 그룹을 모두 수행
        for group in self.param_groups:
            #한 파라미터 그룹의 모든 파라미터 수행
            for p in group["params"]:
                if p.grad is None: #그래디언트가 계산 안됬다면 넘어기기
                    continue
                grad = p.grad.data #p.grad는 torch.Tensor 형태의 객체인데 .data를 통해 tensor값만 가져옴
                if grad.is_sparse: #adam은 희소행렬을 지원 안해서 희소 행렬인 경우 에러 처리
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                #self.state는 딕셔너리이며 p의 옵티마이저 상태(exp_avg, exp_avg_sq, step)를 저장, exp_avg가 그래디언트 m이고 exp_avg_sq가 v임
                state = self.state[p]

                # state가 비어 있다면 초기화
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)      # m (1차 모멘텀)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)   # v (2차 모멘텀)

                # Access hyperparameters from the `group` dictionary.
                #현재 파라미터에서 학습률을 꺼냄
                alpha = group["lr"]

                #1. 1차, 2차 모멘트 업데이트(이것은 학습 전에 설정해놓은 고정값이다.-> 얼마나 오래전 gradient를 기억할지 정함)
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                state["step"] += 1

                #현재 모먼트 상태 가져오기
                #이 모먼트 값들은 학습에 따른 정보가 저장된 값이라고 보면 된다.
                exp_avg = state["exp_avg"] #현재 1차 모먼트 m
                exp_avg_sq = state["exp_avg_sq"] #현재 2차 모먼트 v

                #adam에서 모먼트 계산 수식이 m = b1*m + (1-b1)*m 인데 이를 pytorch의 in place 연산 함수로 계산한 것
                #mul_은 곱하는 연산이고 add_는 grad 텐서에 alpha를 곱하고 더하는 연산이다.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                #모먼트2의 연산식이 v = b2*v + (1-b2)*g^2
                #mul_은 곱하는 연산이고 addcmul_는 (grad*grad)*value를 더하는 연산이다.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #2. 바이어스(편향) 보정
                #https://arxiv.org/abs/1412.6980에 제공된 논문에 의하면 바이어스 보정 식이 다음과 같음
                #αt = α * sqrt(1 - β2^t) / (1 - β1^t), θt = θt−1 − αt * mt / (sqrt(vt) + ε), 여기서 a는 기본 학습률, at는 step t일때 학습률
                #θt는 t번째 iteration의 파라미터 값(가중치), θt−1는 직전 파라미터(직전 가중치), mt와 vt는 1차 모먼트와 2차 모먼트 계수이다.
                # ε는 분모가 0이 되는 것을 방지하는 eps
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                alpha_t = alpha * (bias_correction2 ** 0.5) / bias_correction1
                # 파라미터 업데이트(그래디언트 기반 업데이트)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-alpha_t)
                
                #adamW의  weight decay 적용 수식이 θ←θ−α⋅λ⋅θ 인데 그래디언트로 학습하는 것과 별개로 파라미터θ에 직접 λ: weight decay를 빼준다.
                # weight decay (AdamW는 gradient에 적용하지 않고 직접 파라미터에 적용)
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * alpha)

        #closure 인자를 받았다면 재계산된 loss값을 반환해야 하기 때문에 loss를 리턴한다.
        return loss


                # 
                # TODO: AdamW 구현을 완성하시오. 
                #     위의 state 딕셔너리를 사용하여 상태를 읽고 저장하시오.
                #     하이퍼파라미터는 `group` 딕셔너리에서 읽을 수 있다(생성자에 저장된 lr, betas, eps, weight_decay).

                #         이 구현을 완성하기 위해서 해야할 일들:
                #           1. 그래디언트의 1차 모멘트(첫 번째 모멘트)와 2차 모멘트(두 번째 모멘트)를 업데이트.
                #           2. Bias correction을 적용(https://arxiv.org/abs/1412.6980 에 제공된 "efficient version" 사용; 프로젝트 설명의 pseudo-code에도 포함됨).
                #           3. 파라미터(p.data)를 업데이트.
                #           4. 그래디언트 기반의 메인 업데이트 후 weight decay 적용.

                #         자세한 내용은 기본 프로젝트 안내문을 참조할 것.
                
                ### 완성시켜야 할 빈 코드 블록
                #raise NotImplementedError

        #return loss
