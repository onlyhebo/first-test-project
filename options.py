def add_train_args(parser):
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('--src-word-vec-size', type=int, default=500,
                                help='Word Embdding Size for Src')
    group = parser.add_argument_group('Multiprocessing')
    group.add_argument('--num-process', type=int, default=1,
                                help='process num')
    group = parser.add_argument_group('dataset')
    group.add_argument('--src-file-path', type=str, required=True)
    group.add_argument('--tgt-file-path', type=str, required=True)
    group.add_argument('--dataset-seed', type=int, default=1,
                                help='random seed for data itr')
    group = parser.add_argument_group('get_batch')
    group.add_argument('--batch-size', type=int, default=1,
                                help='set batch size for train')
