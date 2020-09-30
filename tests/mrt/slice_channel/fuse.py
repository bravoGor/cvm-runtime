import mxnet as mx
import numpy as np

from mrt.gen.tfm_types import USQuantizer, SBuffer
from mxnet import ndarray as nd, gluon
from mrt import utils
from mrt.tfm_base import N
from mrt.tfm_pass import convert_params_dtype
from os import path

def generate(shp, ctx=mx.cpu()):
    assert len(shp) == 4
    data_np = np.random.uniform(size=shp)
    return nd.array(data_np, ctx=ctx)

def forward(net, data, ctx, olen, baxis):
    data = gluon.utils.split_and_load(
        data, ctx_list=ctx, batch_axis=baxis, even_split=False)
    outs = [net(d) for d in data]
    if olen == 1:
        outs = nd.concatenate(outs)
    else:
        outs = [nd.concatenate([outs[i][j] \
            for i in range(len(outs)) for j in range(olen)])]
    return outs

@N.register_nm("test_fuse")
def test(_symbol, _params):
    quant = USQuantizer()

    # init value
    xshp = (1, 32, 56, 56)
    wshp = (32, 32, 1, 1)
    attr = {
        'layout': 'NCHW',
        'num_filter': '32',
        'dilate': '(1, 1)',
        'num_group': '1',
        'stride': '(1, 1)',
        'no_bias': 'True',
        'kernel': '[1, 1]'
    }
    ctx = mx.cpu()

    # dump graph
    X = mx.sym.var("data")
    W = mx.sym.var("weight")

    # dump params
    data = nd.load(path.expanduser("~/data.npy"))[0]
    weight = nd.load(path.expanduser("~/weight.npy"))[0]
    xn = X.attr('name')
    wn = W.attr('name')

    # kwargs
    params = {W.attr('name'): weight}
    features = {
        xn: quant.sample(data),
        wn: quant.sample(weight),
    }
    graph = {
        xn: X,
        wn: W,
    }
    precs = {
        xn: {'out_key': quant.get_prec(features[xn].get())},
        wn: {},
    }
    buffers = {
        xn: SBuffer(1),
        wn: SBuffer(1),
    }
    shift_bits = 5

    # original model
    ctx = [ctx]
    sym = mx.sym.Convolution(X, W, **attr, name="Convolution")
    sn = sym.attr('name')
    org_graph = gluon.nn.SymbolBlock(sym, [mx.sym.var("data")])
    utils.load_parameters(org_graph, params, ctx=ctx)
    outs = forward(org_graph, data, ctx, len(sym), 0)

    # quantized model
    kwargs = {
        'params': params,
        'features': features,
        'precs': precs,
        'graph': graph,
        'shift_bits': shift_bits,
        'buffers': buffers,
        'oname': sn,
    }
    oprec = 8
    Xq, xprec, xscale = quant.quantize(X, oprec, **kwargs)
    Wq, wprec, wscale = quant.quantize(W, oprec, **kwargs)
    qsym = mx.sym.Convolution(Xq, Wq, **attr)
    qgraph = gluon.nn.SymbolBlock(qsym, [mx.sym.var("data")])
    print(type(qgraph))
    exit()
    # qparams = convert_params_dtype(params, dest_dtype="float32")
    # utils.load_parameters(qgraph, qparams)
    utils.load_parameters(qgraph, params)
    outs = forward(qgraph, data, ctx, len(qsym), 0)

if __name__ == '__main__':
    _symbol, _params = None, None
    test(_symbol, _params)
