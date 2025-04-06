import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
import torch_knn.torch_knn as torch_knn

class KNearestNeighbor(Function):
    @staticmethod
    def forward(ctx, ref, query, k):  # 添加k参数
        ref = ref.float().cuda()
        query = query.float().cuda()
        
        # 保存反向传播所需参数（网页7）
        ctx.save_for_backward(ref, query)
        ctx.k = k
        
        inds = torch.empty(query.shape[0], 1, query.shape[2]).long().cuda()
        torch_knn.knn(ref, query, inds)
        return inds

    @staticmethod
    def backward(ctx, grad_output):
        # 若无需梯度计算，返回与forward输入个数相同的None（网页1）
        return None, None, None  # 对应ref, query, k三个输入的梯度


class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    # 删除实例化步骤
    D, N, M = 128, 100, 1000
    ref = Variable(torch.rand(2, D, N))
    query = Variable(torch.rand(2, D, M))
    
    # 直接通过apply调用（网页4）
    inds = KNearestNeighbor.apply(ref, query, 2)  # 传入k值
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
    #ref = ref.cpu()
    #query = query.cpu()
    print(inds)


if __name__ == '__main__':
  unittest.main()
