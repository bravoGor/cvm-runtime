import mxnet as mx
import os
from mxnet import ndarray as nd
from os import path

from mrt.sym_utils import topo_sort, sym_iter, get_entry_id
from mrt.tfm_pass import infer_shape

if __name__ == "__main__":
    sym_file = path.expanduser("~/mrt_model/tf_mobilenet_v1_0.25_224_lite.json")
    prm_file = path.expanduser("~/mrt_model/tf_mobilenet_v1_0.25_224_lite.params")
    batch_size = 16
    input_shape = (batch_size, 3, 224, 224)
    sym = mx.sym.load(sym_file)
    prm = nd.load(prm_file)
    infer_shapes = infer_shape(sym, prm, input_shape=input_shape)
    for s in topo_sort(sym):
        name, op_name = s.attr('name'), s.attr('op_name')
        if op_name == "Convolution":
            oshp = infer_shapes[name][get_entry_id(s)]
            attrs, childs = s.list_attr(), sym_iter(s.get_children())
            num_group = eval(attrs['num_group'])
            if num_group > 1:
                continue
            cshps = [infer_shapes[c.attr('name')][get_entry_id(c)] for c in childs]
            assert cshps[0][1] == cshps[1][1]
            O, C = cshps[1][:2]
            nshp = tuple(list(oshp[:1]) + [O*C] + list(oshp[2:]))
            print(
                "oshp: %s, xshp: %s, wshp: %s, new op shp: %s" % \
                (oshp, cshps[0], cshps[1], nshp))
