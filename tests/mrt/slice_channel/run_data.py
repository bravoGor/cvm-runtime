import mxnet as mx

from mxnet import ndarray as nd, gluon
from os import path
from mrt import cvm_op
from mrt import sim_quant_helper as sim

if __name__ == '__main__':
    sym = mx.sym.load(path.expanduser('~/ryt.json'))
    params = nd.load(path.expanduser('~/ryt.params'))
    inputs_ext = {
        'data': {
            'scale': 48.23630979919249,
            'target_bit': 8,
        }
    }
    # TODO(ryt): run int data
    dataset = ds.DS_REG[ds_name](shp, root=dataset_dir)
    data_iter_func = dataset.iter_func()
    data = gluon.utils.split_and_load(
        data, ctx_list=ctx, batch_axis=baxis,
        even_split=False)
    data = sim.load_real_data(data, 'data', inputs_ext)
