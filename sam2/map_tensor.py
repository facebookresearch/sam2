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

def wrap_dim(i, dim):
    if i < 0:
        return dim + i
    return i

def unwrap(t):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return t.elems
    else:
        return t

def unwrap_i(t, i):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return t.elems[i]
    else:
        return t

def unwrap_fn(t, fn):
    if isinstance(t, MapTensor):
        with no_dispatch():
            return fn(t.elems)
    else:
        return None

def wrap(t):
    if isinstance(t, torch.Tensor):
        return MapTensor(t)
    else:
        return t

def ops_impl(cls, func, types, args, kwargs=None):

    unwrapped_args = tree_map(unwrap, args)
    unwrapped_kwargs = tree_map(unwrap, kwargs)

    if func == torch.ops.aten.native_layer_norm.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
        norm_res = func(*unwrapped_args)
        assert len(norm_res) == 3
        return tuple(wrap(a) for a in norm_res)

    # TODO: I guess if being added against something higher dim
    # we should increase dim overall?
    if func == torch.ops.aten.add.Tensor:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        # print("unwrapped_args")
        # print([type(a) for a in unwrapped_args])
        if not isinstance(args[0], MapTensor) and isinstance(args[1], MapTensor):
            if args[0].dim() == (args[1].dim() + 1):
                return NotImplemented
                # return wrap(func(unwrapped_args[0], unwrapped_args[1].unsqueeze(1)))
            # print("args[0].dim(): ", args[0].dim())
            # print("args[1].dim(): ", args[1].dim())
            # print("type(args[0]): ", type(args[0]))
            # print("type(args[1]): ", type(args[1]))
            # TODO: THIS GETS CALLED???
            return NotImplemented
        pass

    if func in [torch.ops.aten.cat.default, torch.ops.aten.stack.default]:
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
        new_args = []
        for a in unwrapped_args[0]:
            if a.dim() == dim:
                new_args.append(a)
            else:
                assert a.dim() + 1 == dim
                new_args.append(a.unsqueeze(0).expand((size[0],) + a.size()))
        return wrap(func(new_args, wrap_dim(unwrapped_args[1], dim - 1) + 1))

    if func == torch.ops.aten.select.int:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        return wrap(func(unwrapped_args[0], unwrapped_args[1] + 1, unwrapped_args[2]))

    if func == torch.ops.aten.slice.Tensor:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 4, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         wrap_dim(unwrapped_args[1], dim - 1) + 1,
                         unwrapped_args[2],
                         unwrapped_args[3]))

    if func == torch.ops.aten.mean.dim:
        # TODO: THIS MIGHT BE WRONG
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        assert len(unwrapped_args[1]) == 1
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         [wrap_dim(unwrapped_args[1][0], dim - 1) + 1],
                         unwrapped_args[2]))

    view_ops = [torch.ops.aten._unsafe_view.default,
                torch.ops.aten.expand.default]
    if func in view_ops:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        input_size = unwrapped_args[0].size()
        bigger_size = list(input_size[:1]) + unwrapped_args[1]
        return wrap(func(unwrapped_args[0], bigger_size))

    if func is torch.ops.aten.view.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        input_size = unwrapped_args[0].size()
        bigger_size = list(input_size[:1]) + unwrapped_args[1]
        return wrap(unwrapped_args[0].reshape(bigger_size))

    if func in [torch.ops.aten.mm.default, torch.ops.aten.bmm.default]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        return wrap(torch.matmul(*unwrapped_args))

    if func in [torch.ops.aten.unsqueeze.default]:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 2, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        new_i = unwrapped_args[1]
        if new_i >= 0:
            new_i += 1
        return wrap(func(unwrapped_args[0], new_i))

    if func == torch.ops.aten.addmm.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        return wrap(torch.matmul(unwrapped_args[1], unwrapped_args[2]) + unwrapped_args[0])

    if func == torch.ops.aten.convolution.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 9, f"args: {unwrapped_args}"
        a = unwrapped_args[0]
        # print("0 a.size(): ", a.size())
        a = unwrapped_args[0].flatten(0, 1)
        # print("1 a.size(): ", a.size())
        # TODO: It's scary that this .contiguous seems necessary, but I guess we're below composite conv
        # which might expected contiguous output
        resa = func(*((a,) + unwrapped_args[1:])).contiguous()
        # print("0 resa.size(): ", resa.size())
        resb = resa.view((unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:])
        # print("1 resb.size(): ", resb.size())
        res_0 = func(*((unwrapped_args[0][0],) + unwrapped_args[1:]))
        if not torch.allclose(resb[0], res_0):
            print("139203")
            import pdb; pdb.set_trace()
            pass
        return wrap(resb)

    if func == torch.ops.aten.upsample_bilinear2d.default:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        a = unwrapped_args[0]
        # print("0 a.size(): ", a.size())
        a = unwrapped_args[0].flatten(0, 1)
        # print("1 a.size(): ", a.size())
        # TODO: It's scary that this .contiguous seems necessary, but I guess we're below composite conv
        # which might expected contiguous output
        resa = func(*((a,) + unwrapped_args[1:])).contiguous()
        # print("0 resa.size(): ", resa.size())
        resb = resa.view((unwrapped_args[0].size(0), unwrapped_args[0].size(1)) + resa.size()[1:])
        # print("1 resb.size(): ", resb.size())
        res_0 = func(*((unwrapped_args[0][0],) + unwrapped_args[1:]))
        if not torch.allclose(resb[0], res_0):
            print("139203")
            import pdb; pdb.set_trace()
            pass
        return wrap(resb)

    if func == torch.ops.aten.transpose.int:
        assert len(unwrapped_kwargs) == 0
        assert len(unwrapped_args) == 3, f"args: {unwrapped_args}"
        dim = unwrapped_args[0].dim()
        return wrap(func(unwrapped_args[0],
                         wrap_dim(unwrapped_args[1], dim - 1) + 1,
                         wrap_dim(unwrapped_args[2], dim - 1) + 1))

    if func == torch.ops.aten._scaled_dot_product_efficient_attention.default:
        assert len(args) == 5
        if all(isinstance(a, MapTensor) for a in args[:3]):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            sdpa_res = wrap(func(unwrapped_args[0].flatten(0, 1),
                                 unwrapped_args[1].flatten(0, 1),
                                 unwrapped_args[2].flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if isinstance(args[0], MapTensor) and not any(isinstance(a, MapTensor) for a in args[1:]):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 5
            assert unwrapped_args[1].dim() == 4
            assert unwrapped_args[2].dim() == 4
            a0 = unwrapped_args[0]
            a1_size = unwrapped_args[1].size()
            a1 = unwrapped_args[1].unsqueeze(0).expand((a0.size(0),) + a1_size)
            a2 = unwrapped_args[2].unsqueeze(0).expand((a0.size(0),) + a1_size)
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and (not isinstance(args[2], MapTensor))):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 4
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[0].size()[1:])
            a2 = unwrapped_args[2].unsqueeze(0).expand((a1_size[0],) + unwrapped_args[2].size()[1:])
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view(unwrapped_args[0].size())),) + sdpa_res[1:]
        if ((not isinstance(args[0], MapTensor)) and isinstance(args[1], MapTensor) and isinstance(args[2], MapTensor)):
            assert len(unwrapped_kwargs) == 0
            assert len(unwrapped_args) == 5, f"args: {unwrapped_args}"
            assert unwrapped_args[0].dim() == 4
            assert unwrapped_args[1].dim() == 5
            assert unwrapped_args[2].dim() == 5
            a0_size = unwrapped_args[0].size()
            a1_size = unwrapped_args[1].size()
            a0 = unwrapped_args[0].unsqueeze(0).expand((a1_size[0],) + a0_size)
            a1 = unwrapped_args[1]
            a2 = unwrapped_args[2]
            sdpa_res = wrap(func(a0.flatten(0, 1),
                                 a1.flatten(0, 1),
                                 a2.flatten(0, 1),
                                 unwrapped_args[3],
                                 unwrapped_args[4]))
            return (wrap(sdpa_res[0].view((a1_size[0],) + a0_size)),) + sdpa_res[1:]
        return NotImplemented


    res = wrap(func(*unwrapped_args, **unwrapped_kwargs))
    # import sys; sys.exit(1)
    return res

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
        # print("func: ", func)
        res = ops_impl(cls, func, types, args, kwargs)
        if isinstance(res, torch.Tensor):
            unwrapped_args_0 = tree_map(lambda x: unwrap_i(x, 0), args)
            unwrapped_kwargs_0 = tree_map(lambda x: unwrap_i(x, 0), kwargs)
            if func == torch.ops.aten.view.default:
                res_0 = torch.ops.aten.reshape.default(*unwrapped_args_0, **unwrapped_kwargs_0)
            else:
                res_0 = func(*unwrapped_args_0, **unwrapped_kwargs_0)
            if res.elems[0].size() != res_0.size():
                import pdb; pdb.set_trace()
                print("02390")
            if not torch.allclose(res.elems[0], res_0, atol=1e-3, rtol=1e-3):
                import pdb; pdb.set_trace()
                print("SDJFKL")
        else:
            pass
            # print("res got type: ", type(res))
        return res

    __torch_function__ = torch._C._disabled_torch_function_impl

# ts is a higher dim Tensor
def to_map_tensor(ts: torch.Tensor):
    return MapTensor(ts)
