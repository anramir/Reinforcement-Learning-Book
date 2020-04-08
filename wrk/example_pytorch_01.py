import torch

if __name__ == "__main__":
    ONES = torch.tensor([1, 2]).to('cuda')
    print(ONES)
