import torch


@torch.no_grad()
def sub_metric_log_dict(sub_metric_name, sub_metric):
    return {
        f'{sub_metric_name}_{func_name}': func(sub_metric)
        for func_name, func in (
            ('min', torch.min),
            ('max', torch.max),
            ('mean', torch.mean),
            ('median', torch.median),
            ('std', torch.std)
        )
    }
