import torch
import torch_scatter
# ours
def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    # print(input.shape)
    # print(index)
    # print(index.unsqueeze(1).shape)
    # print(index.unsqueeze(1).expand(-1, input.size(1)).shape)
    # print(index.unsqueeze(1).expand(-1, input.size(1)))
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # scatter_add_ demo
    # import torch
    # # 创建一个空的张量作为目标张量
    # size = (4, 4)
    # input_tensor = torch.zeros(size)
    # # 创建索引张量和源张量
    # index = torch.tensor([[1,1,1,1],
    #                     [0,0,0,0],
    #                     [1,1,1,1]])

    # source = torch.tensor([[1.0, 2.0, 3.0, 4.0],
    #                     [5.0, 6.0, 7.0, 8.0],
    #                     [9.0, 10.0, 11.0, 12.0]])
    # # 在指定的维度上使用 scatter_add_
    # dim = 0
    # input_tensor.scatter_add_(dim, index, source)
    # # 打印结果 scatter_add_相当于把所有同标签的embedding加起来存放
    # print(input_tensor)
    # tensor([[ 5.,  6.,  7.,  8.],
    #         [10., 12., 14., 16.],
    #         [ 0.,  0.,  0.,  0.],
    #         [ 0.,  0.,  0.,  0.]])


    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.

    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]
    return c, class_counts


def distance2center(input,center):
    n = input.size(0)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)

    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance