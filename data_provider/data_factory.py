import copy
import warnings

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from utils.reproducibility import (
    DeterministicEpochSampler,
    isolated_rng,
    make_torch_generator,
    resolve_seed_bundle,
    seed_worker,
    stable_json_hash,
)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


class TFTDataReproducibilityWarning(UserWarning):
    """A pre-versioned native-TFT caller is using the legacy loader policy."""


def _normalized_split(flag):
    return str(flag).strip().lower()


def _legacy_shuffle(flag):
    # Preserve the historical contract exactly for non-TFT and semantics-v1:
    # only test/TEST is ordered; validation and other flags are shuffled.
    return _normalized_split(flag) != 'test'


def _resolve_tft_v2_loader_seeds(args):
    if getattr(args, 'model', None) != 'TemporalFusionTransformer':
        return None

    semantics = getattr(args, 'tft_extension_semantics_version', None)
    if semantics is None:
        warnings.warn(
            "TemporalFusionTransformer caller has no SR01 semantics/seed fields; "
            "preserving the legacy DataLoader policy.",
            TFTDataReproducibilityWarning,
            stacklevel=3,
        )
        return None
    if int(semantics) != 2:
        return None

    bundle = getattr(args, '_seed_bundle', None)
    if bundle is None:
        # Direct callers may omit stream roots; resolve deterministic defaults
        # once from `seed`. Production callers carry `_seed_bundle` plus
        # already-effective args fields, which must never be derived again.
        bundle = resolve_seed_bundle(
            args,
            iteration=int(getattr(args, 'run_index', 0)),
        )
    return {
        'data_order_seed': int(bundle.data_order_seed),
        'worker_seed': int(bundle.worker_seed),
        'construction_seed': int(bundle.training_seed),
    }


def _construct_dataset(factory, loader_seeds, **kwargs):
    if loader_seeds is None:
        return factory(**kwargs)
    # Legacy augmentation calls np.random.seed internally. Isolating the whole
    # constructor prevents that implementation detail from perturbing model or
    # sampler RNG streams while preserving the augmented data it produces.
    construction_seed = loader_seeds['construction_seed']
    dataset_args = copy.copy(kwargs['args'])
    dataset_args.seed = int(construction_seed)
    dataset_args.tft_augmentation_seed_source = 'training_seed'
    kwargs['args'] = dataset_args
    with isolated_rng(int(construction_seed)):
        return factory(**kwargs)


def _positional_sample_ids(data_set, local_order):
    if not getattr(data_set, 'positional_sample_ids_supported', True):
        return None
    offset = int(getattr(data_set, 'positional_sample_id_start', 0))
    positions = [offset + int(index) for index in local_order]
    prefix = getattr(data_set, 'positional_sample_id_prefix', None)
    if prefix:
        return [f"{prefix}:{position}" for position in positions]
    return positions


def _attach_loader_metadata(
    data_loader,
    data_set,
    loader_seeds,
    flag,
    sampler_policy,
    local_order=None,
):
    fold_manifest = getattr(data_set, 'fold_manifest', None)
    fold_manifest_hash = getattr(data_set, 'fold_manifest_hash', None)
    if fold_manifest is not None and fold_manifest_hash is None:
        fold_manifest_hash = stable_json_hash(fold_manifest)

    epoch0_order_hash = None
    first_batch_index_hash = None
    if local_order is not None:
        positional_ids = _positional_sample_ids(data_set, local_order)
        if positional_ids is not None:
            epoch0_order_hash = stable_json_hash(positional_ids)
            first_batch_ids = positional_ids[:int(data_loader.batch_size)]
            if first_batch_ids:
                first_batch_index_hash = stable_json_hash(first_batch_ids)

    deterministic = loader_seeds is not None and local_order is not None
    data_loader.reproducibility_metadata = {
        'schema_version': 1,
        'split': _normalized_split(flag),
        'sampler_policy': sampler_policy,
        'fold_manifest': copy.deepcopy(fold_manifest),
        'fold_manifest_hash': fold_manifest_hash,
        'epoch0_order_hash': epoch0_order_hash,
        'first_batch_index_hash': first_batch_index_hash,
        'data_order_seed': loader_seeds['data_order_seed'] if deterministic else None,
        'worker_seed': loader_seeds['worker_seed'] if deterministic else None,
    }


def _build_data_loader(
    data_set,
    args,
    flag,
    batch_size,
    drop_last,
    loader_seeds,
    collate=None,
):
    common = {
        'dataset': data_set,
        'batch_size': batch_size,
        'num_workers': args.num_workers,
        'drop_last': drop_last,
    }
    if collate is not None:
        common['collate_fn'] = collate

    if loader_seeds is None:
        shuffle = _legacy_shuffle(flag)
        data_loader = DataLoader(shuffle=shuffle, **common)
        _attach_loader_metadata(
            data_loader,
            data_set,
            loader_seeds,
            flag,
            'legacy_global_rng_shuffle' if shuffle else 'legacy_ordered',
        )
        return data_loader

    is_train = _normalized_split(flag) == 'train'
    sampler = DeterministicEpochSampler(
        data_set,
        seed=loader_seeds['data_order_seed'],
        shuffle=is_train,
    )
    worker_generator = make_torch_generator(loader_seeds['worker_seed'])
    data_loader = DataLoader(
        sampler=sampler,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        **common,
    )
    sampler.set_epoch(0)
    epoch0_order = list(iter(sampler))
    _attach_loader_metadata(
        data_loader,
        data_set,
        loader_seeds,
        flag,
        'deterministic_epoch_shuffle' if is_train else 'ordered',
        local_order=epoch0_order,
    )
    return data_loader


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    loader_seeds = _resolve_tft_v2_loader_seeds(args)

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = _construct_dataset(
            Data,
            loader_seeds,
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = _build_data_loader(
            data_set, args, flag, batch_size, drop_last, loader_seeds
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = _construct_dataset(
            Data,
            loader_seeds,
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = _build_data_loader(
            data_set,
            args,
            flag,
            batch_size,
            drop_last,
            loader_seeds,
            collate=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = _construct_dataset(
            Data,
            loader_seeds,
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = _build_data_loader(
            data_set, args, flag, batch_size, drop_last, loader_seeds
        )
        return data_set, data_loader
