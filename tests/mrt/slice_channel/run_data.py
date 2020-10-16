import mxnet as mx

from mxnet import ndarray as nd, gluon
from os import path
from mrt import cvm_op
from mrt import sim_quant_helper as sim
from mrt import dataset as ds
from mrt.tfm_pass import convert_params_dtype
from mrt import utils
from mrt.sym_utils import topo_sort, topo_visit_transformer,
    sym_iter, is_inputs, get_entry_id, get_nd_op

def run_model(symbol, params, data, **kwargs):
    _, deps = topo_sort(symbol, with_deps=True)
    th_dict, out_cache = {}, {}
    ctx = kwargs.get('ctx', mx.cpu())
    nparams = convert_params_dtype(params, src_dtypes="float64",
            dest_dtype="float32")
    data = data.astype('float32')

    def _impl(op, params, graph, **kwargs):
        deps = kwargs['deps']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if op_name == 'null':
            out = data if is_inputs(op, params) else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                if name not in deps[n]:
                    # for op like: op = broadcast_mul(X, X)
                    # `cinfos` will have duplicate entries
                    # avoid removing more than once
                    continue
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]

    topo_visit_transformer(symbol, nparams, _impl, deps=deps, data=data, **kwargs)
    out_cache.clear()

    return th_dict

if __name__ == '__main__':
    # dataset configuration
    ds_name = "imagenet"
    dataset_dir = "/home/ryt/.mxnet/datasets"
    shp = [15, 3, 224, 224]
    dataset = ds.DS_REG[ds_name](shp, root=dataset_dir)

    # load raw data
    data_iter_func = dataset.iter_func()
    data, _ = data_iter_func()

    # load real data
    inputs_ext = {
        'data': {
            'scale': 48.23630979919249,
            'target_bit': 8,
        }
    }
    data = sim.load_real_data(data, 'data', inputs_ext)
    print(type(data))

    # # split and load data
    # ctx_list = [mx.gpu(0)]
    # baxis = 0
    # data = gluon.utils.split_and_load(
        # data, ctx_list=ctx_list, batch_axis=baxis,
        # even_split=False)

    # remove params names
    # sym = mx.sym.load(path.expanduser('~/ryt.json'))
    # params = nd.load(path.expanduser('~/ryt.params'))
    sym = mx.sym.load(path.expanduser('~/mrt_model/tf_mobilenet_v1_0.25_224_lite.mrt.quantize.json'))
    params = nd.load(path.expanduser('~/mrt_model/tf_mobilenet_v1_0.25_224_lite.mrt.quantize.params'))
    # wnames = {s.attr('name') for s in topo_sort(sym)}
    # rnames = {k for k in params if k not in wnames}
    # for k in rnames:
        # del params[k]
    ctx = mx.gpu(1)
    run_model(sym, params, data, ctx=ctx)
