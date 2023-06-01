import collections

__all__ = ["flatten_dict"]


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                if isinstance(v[0], dict):
                    items.extend(flatten_list(v, new_key, sep=sep))
                    continue
            items.append((new_key, v))
    return dict(items)
