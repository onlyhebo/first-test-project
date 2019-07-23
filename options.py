def add_train_args(parser):
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('--src-word-vec-size', type=int, default=200,
                                help='Word Embdding Size for Src')
    group = parser.add_argument_group('Multiprocessing')
    group.add_argument('--num-process', type=int, default=1,
                                help='process num')
    group = parser.add_argument_group('dataset')
    group.add_argument('--src-train-path', type=str,  default='data/train_src.txt')
    group.add_argument('--tgt-train-path', type=str,  default='data/train_tgt.txt')
    group.add_argument('--src-test-path', type=str, default='data/test_src.txt')
    group.add_argument('--tgt-test-path', type=str, default='data/test_tgt.txt')
    group.add_argument('--seed', type=int, default=1,
                                help='random seed for data itr')
    group = parser.add_argument_group('get_batch')
    group.add_argument('--batch-size', type=int, default=200,
                                help='set batch size for train')
    group = parser.add_argument_group('Model')
    group.add_argument('--hidden-size', type=int, default=256,
                                help='LSTM hidden size')
    group.add_argument('--lstm-dropout', type=float, default=0.5)
    group.add_argument('--embed-dropout', type=float, default=0.6)
    group.add_argument('--final-dropout', type=float, default=0.5)
    group.add_argument('--epochs', type=int, default=10)
    group.add_argument('--clip-norm', type=float, default=5)
