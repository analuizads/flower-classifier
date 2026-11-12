import torch

print("Versão do PyTorch:", torch.__version__)

GPU = torch.cuda.is_available()
if GPU:
    print("GPU ok")
    print(torch.cuda.get_device_name(0))
else:
    print("GPU NÃO encontrada - vai rodar no CPU")