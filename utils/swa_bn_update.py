from tqdm import tqdm
import torch
def update_bn(loader, model, device=None, tune_bert=False):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if tune_bert:
            seqs, lbl_seqs, _, glbl_lbls = input
            seq_lengths = [len(s) for s in seqs]
            seqs = [" ".join(r_ for r_ in s) for s in seqs]
            inputs = model.module.tokenizer.batch_encode_plus(seqs,
                                                       add_special_tokens=model.module.hparams.special_tokens,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=model.module.hparams.max_length)
            inputs['targets'] = lbl_seqs
            inputs['seq_lengths'] = seq_lengths
            model(**inputs)
        else:
            seqs, lbl_seqs, _, glbl_lbls = input
            model(seqs, lbl_seqs)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)