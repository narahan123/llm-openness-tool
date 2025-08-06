import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO



model_name = "skt/A.X-4.0-VL-Light"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device='cuda')
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

url = "https://huggingface.co/skt/A.X-4.0-VL-Light/resolve/main/assets/image.png"
# 이미지 출처: 국가유산포털 (https://www.heritage.go.kr/unisearch/images/national_treasure/thumb/2021042017434700.JPG)

response = requests.get(url)
response.raise_for_status()
image = Image.open(BytesIO(response.content))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "이미지에 대해서 설명해줘."},
        ],
    }
]

inputs = processor(
    images=[image],
    conversations=[messages],
    padding=True,
    return_tensors="pt",
).to("cuda")

# Decoding parameters (top_p, temperature, top_k, repetition_penalty) should be tuned depending on the generation task.
generation_kwargs = {
    "max_new_tokens": 256,
    "top_p": 0.8,
    "temperature": 0.5,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "do_sample": True,
}
generated_ids = model.generate(**inputs, **generation_kwargs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(response[0])
"""
숭례문은 대한민국 서울에 위치한 국보 제1호로, 조선 시대에 건축된 목조 건축물이다. 이 문은 서울의 남쪽 대문으로, 전통적인 한국 건축 양식을 보여준다. 두 층으로 이루어진 이 문은 기와지붕을 얹고 있으며, 지붕의 곡선이 아름답게 표현되어 있다. 문 아래에는 아치형의 출입구가 있으며, 그 주위로는 견고한 석재로 쌓은 성벽이 이어져 있다. 배경에는 현대적인 고층 빌딩들이 자리잡고 있어, 전통과 현대가 공존하는 서울의 모습을 잘 나타낸다. 숭례문은 역사적, 문화적 가치가 높아 많은 관광객들이 찾는 명소이다.
"""

