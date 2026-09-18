from torch.utils._sympy import printers
from transformers.models import vision_text_dual_encoder


class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)         #数据
        self.grad = 0.0                 #梯度
        self._prev = set(_children)     #父节点
        self._backward = lambda: None   #记录如何把梯度传给上一级

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        # 加法： out = self + other
        # d(out)/d(self) = 1, d(out)/d(other) = 1
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        # 乘法： out = self * other
        # d(out)/d(self) = other , d(out)/d(other) = self
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)
    c = a * b  # c.data = 6.0
    d = c + a  # d.data = 8.0
    d.backward()  # 自动求导！
    print(a.grad)
    print(b.grad)
    print(c.grad)