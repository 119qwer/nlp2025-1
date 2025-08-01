from torch import dtype

from config import PretrainedConfig
from utils import *


class GPTPreTrainedModel(nn.Module):

    #~~config와 모델의 이름(name_or_path)를 설정
  def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    super().__init__()
    self.config = config
    self.name_or_path = config.name_or_path

  def init_weights(self):
    # 가중치 초기화
    self.apply(self._init_weights)

    #~~가중치 초기화에 사용하는 함로로
  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)): #~~정규분포 초기화
      # 초기화를 위해 truncated_normal을 사용하는 TF 버전과 약간 차이가 있다.
      # (참고) https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm): #~~weight는 1, bias는 0으로 초기화
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

    #~~파라미터의 데이터타입을 반환하는 함수, get_parameter_dtype는 utils.py에 정의
  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)
  