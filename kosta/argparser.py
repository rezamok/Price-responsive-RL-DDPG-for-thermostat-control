# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:31:13 2021

@author: kosta
"""
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathDB', type=str, default=hp.dataDir,
                          help='Path to the directory containing the '
                          'data.')
    group_db.add_argument('--file_extension', type=str, default=hp.ext,
                          help="Extension of the audio files in the dataset.")
    group_db.add_argument('--pathTrain', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'training sequences.')
    group_db.add_argument('--pathVal', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'validation sequences.')
    group_db.add_argument('--n_process_loader', type=int, default=hp.num_workers,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--max_size_loaded', type=int, default=hp.max_size_loaded,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    group_supervised = parser.add_argument_group(
        'Supervised mode (depreciated)')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='(Depreciated) Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the speaker classification.')
    group_supervised.add_argument('--pathPhone', type=str, default=None,
                                  help='(Supervised mode only) Path to a .txt '
                                  'containing the phone labels of the dataset. If given '
                                  'and --supervised, will train the model using a '
                                  'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=hp.resultsDir,
                            help="Path of the output directory.")
    group_save.add_argument('--logging_step', type=int, default=hp.logging_step)
    group_save.add_argument('--save_step', type=int, default=hp.save_step,
                            help="Frequency (in epochs) at which a checkpoint "
                            "should be saved")

    group_load = parser.add_argument_group('Load')
    group_load.add_argument('--load', type=str, default=None, nargs='*',
                            help="Load an exsiting checkpoint. Should give a path "
                            "to a .pt file. The directory containing the file to "
                            "load should also have a 'checkpoint.logs' and a "
                            "'checkpoint.args'")
    group_load.add_argument('--loadCriterion', action='store_true',
                            help="If --load is activated, load the state of the "
                            "training criterion as well as the state of the "
                            "feature network (encoder + AR)")
    group_load.add_argument('--restart', action='store_true',
                            help="If any checkpoint is found, ignore it and "
                            "restart the training from scratch.")

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=hp.nGPU,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=hp.batchSizeGPU,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    args = parser.parse_args(argv)

    if args.pathDB is None and (args.pathCheckpoint is None or args.restart):
        parser.print_help()
        print("Either provides an input dataset or a checkpoint to load")
        sys.exit()

    if args.pathCheckpoint is not None:
        args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)

    if args.load is not None:
        args.load = [os.path.abspath(x) for x in args.load]

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")

    if args.arMode == 'no_ar':
        args.hiddenGar = args.hiddenEncoder
    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)