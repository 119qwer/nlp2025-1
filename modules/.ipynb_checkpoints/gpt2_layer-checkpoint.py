from torch import nn

import torch.nn.functional as F

#from modules.attention import CausalSelfAttention
#같은 디렉토리안에 있는 데 modules.attention라고 하면 인식이 안되서 수정했다.
from modules.attention import CausalSelfAttention

#입력 시퀀스를 받아서 Self-Attention과 FeedForward를 거치게 하는 클래스, init으로 초기화하고 foward함수를 사용하여 Self-Attention과 FeedForward를 적용
class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
      #CausalSelfAttention는 문맥을 이해하기 위한 self attention 함수인데 자신 이후의 토큰은 보지 못하도록 제한한다.
    self.self_attention = CausalSelfAttention(config)
      
    # Add-norm for multi-head attention.
      #다양한 문맥정보 추출을 위해 multi head attention을 하는 데 이 head마다의 attention 결과를 통합하는 게 attention_dense이다.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
      #self attention 결과를 정규화 수행하는 부분, nn.LayerNorm은 평균이 0, 분산이 1이 되도록 정규화해준다.
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
      #과적합을 방지하기 위해서 훈련데이터 일부분을 무작위로 0으로 만들고 다른 것들을 높여서 맞춰주는 함수(훈련시에만 적용)
      #hidden_dropout_prob가 0.1이면 10%를 무작위로 0으로 만든고 나머지는 1.11배
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
      
    # Feed forward.(ffn의 첫번째 선형변환)
      
      #입력 벡터(hidden_size 차원)를 더 넓은 차원(intermediate_size)으로 선형 확장하는 코드, 더 풍부한 비선형 표현을 학습하기 위해 잠깐 차원을 키우는 것
      #나중에 가중치 W를 곱해주면서 원래 차원만큼 다시 작아질거임
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
      #F.gelu는 GELU함수로 Relu와 유사하지만 더 부드럽게 통과시킨다. 입력이 작은 것은 더 약하게, 클수록 그대로 통과시키는 필터(더 유의미한 것을 통과시키는 필터)
    self.interm_af = F.gelu
    # Add-norm for feed forward.(ffn의 두번째 선형변환)
      #차원을 늘리고 GELU를 활성화하고 다시 차원을 줄이는 데 이 코드가 차원을 줄이는 코드
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
      
      #ffn 끝에서 사용하는 layer Normalization(정규화)이다. 이전 정규화와 같이 평균을 0, 분산은 1로 맞추어준다.
      #이전 attention의 정규화도 마찬가지로 이건 정규화 후 forward()에서 Residual을 적용할것임
      #Residual은 입력값을 처리결과(Layer)에 더해서 전달하는 방식이다. ex) x = x + self.attention_dropout(self.attention_dense(attn_output))
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
      #마찬가지로 과적합을 막기 위한 dropout
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: forward() 함수를 위한 이 helper 메서드를 구현하시오:
      - 이 함수는 multi-head attention layer와 feed forward layer 이후에 적용된다.
      - GPT-2 layer는 각 sublayer의 변환된 출력에 드롭아웃을 적용한 후, 이를 sublayer 입력에 더한다. 
        이 함수에서는 Layer Normalization을 적용하지 않는다.
    """
    out = dense_layer(output)
    out = dropout(out)
    return input + out  # Residual 연결 (정규화는 안 함)


    #초기화에서 dropout까지 했으니 이후 Residual->정규화를 하면 된다.
    #init에서의 초기화를 사용하는 데 예를 들어
    #self.interm_dense = nn.Linear(768, 3072)
    #self.interm_af = F.gelu
    #self.out_dense = nn.Linear(3072, 768)
    #라고 init에서 했으면
    #x = self.interm_dense(x)     # 차원 확장 (768 → 3072)
    #x = self.interm_af(x)        # 활성화 함수 (GELU)
    #x = self.out_dense(x)        # 차원 축소 (3072 → 768)
    #라고 foward에서 사용할거임
    
  def forward(self, hidden_states, attention_mask):
    # -------------------------------
    # 1. Self-Attention Block
    # -------------------------------
    residual = hidden_states  # 🔹 잔차용 복사

    # 🔸 Self-Attention 연산 (즉, self_attention의 forward 함수 실행)
    attention_output = self.self_attention(
        hidden_states,        # Q, K, V
        attention_mask        # mask는 토큰간 주의를 제한히기 위함(원하는 토큰만 보기 위함)
    )

    # 🔸 후처리: 투영(head들의 attention 통합) → 드롭아웃(과적합 방지)
    attention_output = self.add(residual, attention_output, self.attention_dense, self.attention_dropout)
    #attention_output = self.attention_dense(attention_output)
    #attention_output = self.attention_dropout(attention_output)

    # 🔸 Residual + LayerNorm (residual 후에 정규화 하는 것이 원칙, 아니면 residual의 효과가 반감됨)
    hidden_states = self.attention_layer_norm(attention_output)

    # -------------------------------
    # 2. Feed Forward Network Block(Self-Attention을 통해 토큰 간 관계를 학습한 후 ffn으로 비선형적으로 가공)
    # -------------------------------
    residual = hidden_states  # 🔹 다시 잔차용 복사

    # 🔸 FFN 확장 → 활성화 → 축소
    ff_output = self.interm_dense(hidden_states) # 차원 확장
    ff_output = self.interm_af(ff_output)       # GELU함수 적용
    ff_output = self.add(residual, ff_output, self.out_dense, self.out_dropout)
    #ff_output = self.out_dense(ff_output)       #차원 축소
    #ff_output = self.out_dropout(ff_output)     #드랍아웃

    # 🔸 Residual + LayerNorm
    hidden_states = self.out_layer_norm(ff_output)

    # 🔚 최종 출력
    return hidden_states
        
  #def forward(self, hidden_states, attention_mask):
    # """
    # TODO: forward pass의 구현. 고려해야 할 주요 사항은 다음과 같다:
    #   - Multi-head Attention layer(CausalSelfAttention): mask된 입력을 기반으로 self-attention을 계산한다.
    #   - Layer Normalization: Attention layer와 Feed-forward layer 이전에 적용된다.
    #   - Dropout, Residual Connection, Layer Normalization를 적용하시오(self.add() 메서드를 사용)
    #   - Feed-Forward layer: hidden states를 추가로 refine하기 위해 변환을 적용한다.
    # """

    ### 완성시켜야 할 빈 코드 블록
    #raise NotImplementedError

import torch
import torch.nn as nn
import torch.nn.functional as F

# 테스트용 config 클래스 정의
class Config:
    def __init__(self):
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-5
        self.max_position_embeddings = 128
        self.n_head = 12
        # ❗️추가해 주세요:
        self.num_attention_heads = 12  # 또는 self.n_head도 같이 둘 수 있음
        self.max_position_embeddings = 128

# 가짜 causal mask 생성 함수
def generate_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, T, T]

# 테스트용 GPT2Layer 인스턴스 생성
config = Config()
layer = GPT2Layer(config)

# 입력 텐서 생성: [batch, seq_len, hidden_size]
batch_size = 2
seq_len = 16
hidden_size = config.hidden_size
x = torch.randn(batch_size, seq_len, hidden_size)

# 마스크 생성
attention_mask = generate_causal_mask(seq_len)

# forward 실행
with torch.no_grad():
    out = layer(x, attention_mask)

print("입력 shape:", x.shape)
print("출력 shape:", out.shape)

