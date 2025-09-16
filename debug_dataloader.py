from data_provider.data_factory import data_provider
import argparse

if __name__ == '__main__':
    # Create args similar to the training script
    args = argparse.Namespace(
        task_name='long_term_forecast',
        data='ETTh1',
        root_path='./dataset/ETT-small/',
        data_path='ETTh1.csv',
        features='M',
        target='OT',
        seq_len=96,
        label_len=48,
        pred_len=96,
        embed='timeF',
        freq='h',
        batch_size=32,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        enc_in=7,
        dec_in=7,
        c_out=7,
        inverse=False,
        augmentation_ratio=0,
        use_norm=1
    )

    # Get the data loader
    train_set, train_loader = data_provider(args, 'train')

    # Get a sample from the loader
    sample = next(iter(train_loader))

    print(f'Sample type: {type(sample)}')
    print(f'Sample length: {len(sample)}')
    print(f'Sample structure: {[type(x) for x in sample]}')
    print(f'Sample shapes: {[x.shape if hasattr(x, "shape") else len(x) for x in sample]}')

    # Try to unpack and see what happens
    try:
        batch_x, batch_y, batch_x_mark, batch_y_mark = sample
        print('Successfully unpacked 4 values')
        print(f'batch_x shape: {batch_x.shape}')
        print(f'batch_y shape: {batch_y.shape}')
        print(f'batch_x_mark shape: {batch_x_mark.shape}')
        print(f'batch_y_mark shape: {batch_y_mark.shape}')
    except ValueError as e:
        print(f'Unpacking error: {e}')
        print('Detailed content:')
        for idx, item in enumerate(sample):
            print(f'  Item {idx}: type={type(item)}, shape={getattr(item, "shape", "N/A")}')
            if hasattr(item, 'shape') and len(item.shape) <= 2:
                print(f'    Content preview: {item}')
        print('Trying different unpacking...')
        if len(sample) == 3:
            batch_x, batch_y, batch_x_mark = sample
            print('Successfully unpacked 3 values')
        elif len(sample) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, extra = sample
            print('Successfully unpacked 5 values')
            print(f'Extra value type: {type(extra)}, shape: {extra.shape if hasattr(extra, "shape") else len(extra)}')