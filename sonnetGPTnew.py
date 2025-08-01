
'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from datasets import (
    SonnetsDataset,
)
from models.TestGPT import TestGPT
from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

    def __init__(self, args):
        super().__init__()
        self.gpt = TestGPT.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
        for param in self.gpt.parameters():
            param.requires_grad = True  # 여기 true인데 임시로 False로 바꾼 거임@@@@@@

        # 첫번째 방법 : requires_grad를 False로 하고  마지막 출력 레이어만 학습하기(last layer tuning)
        # 두번째 방법 : Transformer 블록 사이에 adapter를 삽입하고 이부분만 학습(Adapter tuning)
        # 세번째 방법 : 가중치 행렬을 rank 분해해서 일부분만 학습(LoRA)
        # 이를 위해서는 일부분만 학습하는 방법을 알아야 함
        # 첫 방법은 데이터가 적을 때 좋고, 두번째 방법은 다양한 스타일에 좋고, 3번째는 성능과 속도가 좋다.
        # 소네트 스타일만 사용할 것이기 때문에 현재는 첫번째와 두번째만 고려 사항이다.

        # 첫번째 방법
        # # 마지막 레이어만 fine-tuning 설정 (gpt2를 보면 gpt_layers 변수에 레이어를 저장하고 있고 거기서 마지막 레이어만 꺼내서 파인튜닝하는 것)
        # for param in self.gpt.gpt_layers[-1].parameters():
        #     param.requires_grad = True

        # 3번째 방법 (LoRa가 좋은 이유는 사전학습된 GPT-2는 너무 큰데 소네트 데이터는 그에 비해 작아서 학습이 안좋은데 저랭크 가중치만 학습하여 효율이 좋음)
        # 기존 가중치를 동결하고 작은 랭크 가중치만 학습할 수 있게 하는 LoRALinear클래스를 만들고 우리가 만든 self_attention클래스의
        # query와 value에 self.query = LoRALinear(config.hidden_size, self.all_head_size, r=8, alpha=32) 와 같이 적용기기

    def forward(self, input_ids, attention_mask):
        ### 완성시켜야 할 빈 코드 블록
        """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
        # 1. 임베딩
        embedding_output = self.gpt.embed(input_ids=input_ids)

        # 2. 트랜스포머 인코딩
        hidden_states = self.gpt.encode(embedding_output, attention_mask=attention_mask)
        hidden_states = self.gpt.final_layer_norm(hidden_states)

        # 3. 로짓 계산 (vocab size로 투사)
        logits = self.gpt.hidden_state_to_token(hidden_states)
        return logits;

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

        for _ in range(max_length):
            # logits을 구하기 위한 forward pass.
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)

    # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    #   #테스트 용 드드
    # for name, param in model.named_parameters():
    #   if 'lora' in name:
    #       print(f"{name}, requires_grad={param.requires_grad}, shape={param.shape}")

    lr = args.lr
    #####수정 부분(아래 한줄 활성화하고 두줄은 지우면 원래대로 됨)
    #optimizer = AdamW(model.parameters(), lr=lr)
    # # 'lora_'라는 이름이 붙은 파라미터만 학습 대상
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = AdamW(lora_params, lr=1e-4, weight_decay=0.0)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
            labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
        print('Generating several output sonnets...')
        model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            print(f'{batch[1]}{output[1]}\n\n')

        # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs - 1}_{args.filepath}', weights_only=False)

    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f'{decoded_output}\n\n'
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(f'{decoded_output}\n\n')

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    # Generation parameters.
    parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
    parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                        default=0.9)

    parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == 'gpt2':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise Exception(f'{args.model_size} is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
    seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
    train(args)
    generate_submission_sonnets(args)
