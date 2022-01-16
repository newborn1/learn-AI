import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,5]])#不要忘了中间也有，

variable = Variable(tensor,requires_grad=True)

print(variable)

print(Variable(torch.FloatTensor([[6,5],[6,9]])))#这是Variable类型，也是tensor

print(isinstance(variable,torch.Tensor))
print(isinstance(variable,Variable))

value = variable.data

print(value)

print(value.numpy())