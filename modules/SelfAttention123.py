import torch
from modules.LoRALinear import LoRALinear
from einops import rearrange
from torch import nn

class SelfAttention123(nn.Module):
  def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # LoRA 버전 Linear 레이어 사용
        self.query = LoRALinear(config.hidden_size, self.all_head_size, r=4, lora_alpha=16)
        self.key = LoRALinear(config.hidden_size, self.all_head_size, r=4, lora_alpha=16)
        self.value = LoRALinear(config.hidden_size, self.all_head_size, r=4, lora_alpha=16)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다.
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며,
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
      key: [batch_size, num_attention_heads, seq_len_k, attention_head_size]
      query: [batch_size, num_attention_heads, seq_len_q, attention_head_size]
      value: [batch_size, num_attention_heads, seq_len_v, attention_head_size]
    """
    ### 완성시켜야 할 빈 코드 블록

    # 1. Scaled Dot-Product 계산 - (Query * Key^T) / sqrt(d_k)
    transposed_key = key.transpose(-1, -2)  # (B, H, D_h, T_k)
    raw_attention_scores = torch.matmul(query, transposed_key)  # (B, H, T_q, T_k)

    # 스케일링
    scaling_factor = torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
    scaled_attention_scores = raw_attention_scores / scaling_factor

    # ✅ causal mask 적용
    seq_len = query.size(-2)
    device = query.device
    dtype = query.dtype

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
    causal_mask = (1.0 - causal_mask) * -10000.0

    # ✅ attention_mask와 함께 적용
    if attention_mask is not None:
        masked_attention_scores = scaled_attention_scores + attention_mask + causal_mask
    else:
        masked_attention_scores = scaled_attention_scores + causal_mask


    # 3. Softmax to get Attention Probabilities
    attention_weights = torch.nn.functional.softmax(masked_attention_scores, dim=-1)

    # 4. Dropout 적용
    attention_weights = self.dropout(attention_weights)

    # 5. Value와의 Weighted Sum
    #   Attention_Weights: (B, H, T_q, T_k)
    #   Value:             (B, H, T_v, D_h)  (여기서 T_k == T_v)
    #   Result (Context):  (B, H, T_q, D_h)
    context_vectors = torch.matmul(attention_weights, value)  # (B, H, T_q, D_h)

    # 6. 헤드들을 합치기
    final_context_vector = rearrange(context_vectors, 'b h t d -> b t (h d)')
    return final_context_vector


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
        