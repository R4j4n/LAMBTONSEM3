import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TextInference(metaclass=Singleton):
    def __init__(self, model_name: str = "NumbersStation/nsql-350M"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = PeftConfig.from_pretrained("Rajan/training_run")
        base_model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")
        self.model = PeftModel.from_pretrained(base_model, "Rajan/training_run").to(
            self.device
        )

    def generate_text(self, input_text: str, max_length: int = 500):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.device
        )
        generated_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
