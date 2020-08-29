import logging
import math
import numpy as np

from mxnet import ndarray as nd
import mxnet as mx
import cvm

from mrt.sym_utils import get_attr, sym_iter, is_params, is_inputs, \
                          nd_array, get_mxnet_op, get_nnvm_op, \
                          nd_const, get_entry_id
from mrt.tfm_utils import get_bit_cnt
from mrt.tfm_base import N, MAX_BIT
from mrt.tfm_pass import OUT_KEY
from .tfm_base import register_pass, register_transformer, Transformer
from .tfm_types import get_quantizer, USQuantizer, UAQuantizer, \
                       FT_TYPE_EXP, AFeature, SBuffer
from .tfm_utils import scale_exp, get_buffer_exp

from mrt import sim_quant_helper as sim
from mrt import sym_utils as sutils
from mrt import tfm_ops as tops


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Flatten")
class Flatten(tops.Flatten, Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("null")
class Null(tops.Null, Transformer):
    def quantize(self, op, **kwargs):
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            ft = kwargs['features'][name]
            assert ft.name == FT_TYPE_EXP
            kwargs['buffers'][name] = get_buffer_exp(
                scale_exp(ft.get(), prec))
            extra_attr = {'precision': str(prec)}
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op


@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Pooling")
class Pooling(tops.Pooling, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(tops.Dropout, Transformer):
    pass


@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("Activation")
class Activation(tops.Activation, Transformer):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(tops.FullyConnected, Transformer):
    def rewrite(self, op, **kwargs):
        op = super().rewrite(op, **kwargs)
        op = separate_bias(op, **kwargs)
        return op

    def quantize(self, op, **kwargs):
        features, buffers = kwargs['features'], kwargs['buffers']
        precs = kwargs['precs']
        cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        assert len(childs) == 2
        xquant_type = cfg_dict[cns[0]]['quant_type']
        wquant_type = cfg_dict[cns[1]]['quant_type']

        if xquant_type == wquant_type == USQuantizer.name:
            op = _quantize_xw(op, **kwargs)
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s" % \
                (op_name, [xquant_type, wquant_type]))

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("Convolution")
class Convolution(tops.Convolution):
    def rewrite(self, op, **kwargs):
        op = super().rewrite(op, **kwargs)
        op = separate_pad(op, **kwargs)
        op = separate_bias(op, **kwargs)
        return op

    def quantize(self, op, **kwargs):
        features, buffers = kwargs['features'], kwargs['buffers']
        precs = kwargs['precs']
        cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        assert len(childs) == 2 and 'pad' not in attr
        xquant_type = cfg_dict[cns[0]]['quant_type']
        wquant_type = cfg_dict[cns[1]]['quant_type']

        if xquant_type == wquant_type == USQuantizer.name:
            op = _quantize_xw(op, **kwargs)
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s" % \
                (op_name, [xquant_type, wquant_type]))

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('Pad')
class Pad(tops.Pad, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_add")
class BroadcastAdd(tops.BroadcastAdd):
    def quantize(self, op, **kwargs):
        params = kwargs['params']
        features = kwargs['features']
        precs = kwargs['precs']
        buffers = kwargs['buffers']

        name = op.attr('name')
        childs = sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        assert all([features[cn].name == FT_TYPE_EXP for cn in cns])
        cfts = [features[cn].get() for cn in cns]

        if cfts[0] == 0 or cfts[1] == 0:
            if cfts[0] == 0 and cfts[1] == 0:
                features[name] = AFeature(0)
                precs[name] = {OUT_KEY: 1}
                buffers[name] = SBuffer(1)
                return op
            cn = cns[1] if cfts[0] == 0 else cns[0]
            bit = get_bit_exp(params[cn]) if cn in params \
                else precs[cn][OUT_KEY]
            buffers[name] = SBuffer(1) if cn in params else buffers[cn]
            precs[name] = {OUT_KEY: bit}
            features[name] = features[cn]
            return op

        return _quantize_scale(op, **kwargs)


@register_pass("rewrite")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("Concat")
class Concat(tops.Concat):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)

def separate_bias(op, **kwargs):
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())

    if len(childs) < 3 or op_name not in \
        [Convolution.op_name, FullyConnected.op_name]:
        return op

    attr['no_bias'] = True
    op = get_mxnet_op(op_name)(
        childs[0], childs[1], **attr, name=N.n(name))
    bn = childs[2].attr('name')
    if op_name == Convolution.op_name:
        assert 'layout' in attr and attr['layout'] == 'NCHW'
        B = mx.sym.expand_dims(childs[2], axis=0, name=N.n('expand_dims'))
        B = mx.sym.expand_dims(B, axis=-1, name=N.n('expand_dims'))
        B = mx.sym.expand_dims(B, axis=-1, name=N.n(bn))
    else:
        B = mx.sym.expand_dims(childs[2], axis=0, name=N.n(bn))
    op = mx.sym.broadcast_add(op, B, name=name)
    return op

def separate_pad(op, **kwargs):
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())

    if op_name not in [Convolution.op_name]:
        return op

    assert 'layout' in attr and attr['layout'] == 'NCHW'
    PH, PW = get_attr(attr, 'pad', (0,0))
    if PH == 0 and PW == 0:
        return op
    del attr['pad']

    childs[0] = mx.sym.pad(
        childs[0], pad_width=(0,0,0,0,PH,PH,PW,PW),
        mode='constant', constant_value=0, name=N.n('pad'))
    op = get_mxnet_op(op_name)(*childs, **attr, name=name)
    return op

def _quantize_scale(op, **kwargs):
    features, precs = kwargs['features'], kwargs['precs']
    buffers, cfg_dict = kwargs['buffers'], kwargs['cfg_dict']
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    assert all([features[cn].name == FT_TYPE_EXP for cn in cns])
    absmax = max([features[cn].get() for cn in cns])
    oprec = kwargs['op_input_precs'][op_name]
    oscale = scale_exp(absmax, oprec)
    buffers[name] = SBuffer(oscale)
    new_childs, cprecs = [], []

    assert all([cfg_dict[cn]['quant_type'] == \
        USQuantizer.name for cn in cns])
    quant = get_quantizer(USQuantizer.name)

    for c in childs:
        c, cprec, _ = quant.quantize(
            c, oprec, oscale=oscale, oname=name, **kwargs)
        cprecs.append(cprec)
        new_childs.append(c)
    op = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    infer_prec = max(cprec) if op_name == Concat.op_name else max(cprecs)+1
    precs[name][OUT_KEY] = infer_prec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, buffers[name].serialize(), cns)
    return op

def _quantize_xw(op, **kwargs):
    features, buffers = kwargs['features'], kwargs['buffers']
    precs = kwargs['precs']
    cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    X, W = childs
    Xquant = get_quantizer(cfg_dict[cns[0]]['quant_type'])
    Wquant = get_quantizer(cfg_dict[cns[1]]['quant_type'])

    oprec = kwargs['op_input_precs'][op_name]
    Xq, xprec, xscale = Xquant.quantize(X, oprec, oname=name, **kwargs)
    Wq, wprec, wscale = Wquant.quantize(W, oprec, oname=name, **kwargs)
    buffers[name] = get_buffer_exp(xscale*wscale)
    op = get_mxnet_op(op_name)(Xq, Wq, **attr, name=name)

    shp = params[cns[1]].shape
    k = int(nd.prod(nd_array(shp[1:])).asscalar())
    kprec = get_bit_cnt(k)
    infer_prec = kprec + xprec + wprec
    precs[name][OUT_KEY] = infer_prec
    return op
