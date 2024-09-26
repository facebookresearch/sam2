import contextlib
import torch
from torch.utils._pytree import tree_map

@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

class MapTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elems):
        elem = elems[0]
        return torch.Tensor._make_wrapper_subclass(cls,
                                                   elem.shape,
                                                   dtype=elem.dtype,
                                                   device=elem.device)

    def __init__(self, elems):
        self.elems = elems

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        print("func: ", func)
        def unwrap(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return t.elems
            else:
                return t

        def wrap(t):
            if isinstance(t, torch.Tensor):
                return MapTensor(t)
            else:
                return t

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        if func == torch.ops.aten.add.Tensor:
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
            print("unwrapped_args")
            print([a.size() if isinstance(a, torch.Tensor) else a for a in unwrapped_args])
            pass

        if func == torch.ops.aten.cat.default:
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
            # TODO: Use MapTensor type for filter
            # First argument's dim
            dim = unwrapped_args[0][0].dim()
            size = unwrapped_args[0][0].size()
            for a in unwrapped_args[0]:
                if a.dim() > dim:
                    dim = a.dim()
                    size = a.size()
            args = []
            for a in unwrapped_args[0]:
                if a.dim() == dim:
                    args.append(a)
                else:
                    assert a.dim() + 1 == dim
                    args.append(a.unsqueeze(0).expand((size[0],) + a.size()))
            return wrap(func(args, unwrapped_args[1] + 1))

        if func == torch.ops.aten.select.int:
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
            return wrap(func(unwrapped_args[0], unwrapped_args[1] + 1, unwrapped_args[2]))

        view_ops = [torch.ops.aten.view.default,
                    torch.ops.aten._unsafe_view.default,
                    torch.ops.aten.expand.default]
        if func in view_ops:
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
            input_size = unwrapped_args[0].size()
            bigger_size = list(input_size[:1]) + unwrapped_args[1]
            return wrap(func(unwrapped_args[0], bigger_size))

        if func == torch.ops.aten.mm.default:
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
            return wrap(torch.matmul(*unwrapped_args))

        res = wrap(func(*unwrapped_args, **unwrapped_kwargs))
        # import sys; sys.exit(1)
        return res

    __torch_function__ = torch._C._disabled_torch_function_impl

# ts is a higher dim Tensor
def to_map_tensor(ts: torch.Tensor):
    return MapTensor(ts)
