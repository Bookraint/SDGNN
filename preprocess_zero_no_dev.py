import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_vast
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM,generate_adj_data_from_grounded_concepts__use_SentBert,generate_adj_data_from_grounded_concepts

input_paths = {
    'vast_zero': {
        'train': './data/vast_zero/vast_train.csv',
        'dev': './data/vast_zero/vast_zero_dev.csv',
        'test': './data/vast_zero/vast_zero_test.csv',
    },
    'vast_few': {
        'train': './data/vast_few/vast_train.csv',
        'dev': './data/vast_few/vast_few_dev.csv',
        'test': './data/vast_few/vast_few_test.csv',
    },
    'vast_all': {
        'train': './data/vast_all/vast_train.csv',
        'dev': './data/vast_all/vast_zero_dev.csv',
        'test': './data/vast_all/vast_zero_test.csv',
    },
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'vast_zero': {
        'statement': {
            'train': './data/vast_zero/statement/train.statement.jsonl',
            'dev': './data/vast_zero/statement/dev.statement.jsonl',
            'test': './data/vast_zero/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/vast_zero/grounded/train.grounded.jsonl',
            'dev': './data/vast_zero/grounded/dev.grounded.jsonl',
            'test': './data/vast_zero/grounded/test.grounded.jsonl',
            'train_1': './data/train_ground_split/train_grounded_1.jsonl',
            'train_2': './data/train_ground_split/train_grounded_2.jsonl',
            'train_3': './data/train_ground_split/train_grounded_3.jsonl',
            'train_4': './data/train_ground_split/train_grounded_4.jsonl',
            'train_5': './data/train_ground_split/train_grounded_5.jsonl',
            'train_6': './data/train_ground_split/train_grounded_6.jsonl',
            'train_7': './data/train_ground_split/train_grounded_7.jsonl',
            'train_8': './data/train_ground_split/train_grounded_8.jsonl',
            'train_9': './data/train_ground_split/train_grounded_9.jsonl',
            'train_10': './data/train_ground_split/train_grounded_10.jsonl',
            'train_11': './data/train_ground_split/train_grounded_11.jsonl',
            'train_12': './data/train_ground_split/train_grounded_12.jsonl',
            'train_13': './data/train_ground_split/train_grounded_13.jsonl',
            'train_14': './data/train_ground_split/train_grounded_14.jsonl',
        },
        'graph': {
            'adj-train': './data/vast_zero/graph/train.graph.adj.pk',
            'adj-dev': './data/vast_zero/graph/dev.graph.adj.pk',
            'adj-test': './data/vast_zero/graph/test.graph.adj.pk',
            # 'adj-train_1':  './data/vast_zero/graph/train_1.graph.adj.pk',
            # 'adj-train_2':  './data/vast_zero/graph/train_2.graph.adj.pk',
            # 'adj-train_3':  './data/vast_zero/graph/train_3.graph.adj.pk',
            # 'adj-train_4':  './data/vast_zero/graph/train_4.graph.adj.pk',
            # 'adj-train_5':  './data/vast_zero/graph/train_5.graph.adj.pk',
            # 'adj-train_6':  './data/vast_zero/graph/train_6.graph.adj.pk',
            # 'adj-train_7':  './data/vast_zero/graph/train_7.graph.adj.pk',
            # 'adj-train_8':  './data/vast_zero/graph/train_8.graph.adj.pk',
            # 'adj-train_9':  './data/vast_zero/graph/train_9.graph.adj.pk',
            # 'adj-train_10': './data/vast_zero/graph/train_10.graph.adj.pk',
            # 'adj-train_11': './data/vast_zero/graph/train_11.graph.adj.pk',
            # 'adj-train_12': './data/vast_zero/graph/train_12.graph.adj.pk',
            # 'adj-train_13': './data/vast_zero/graph/train_13.graph.adj.pk',
            # 'adj-train_14': './data/vast_zero/graph/train_14.graph.adj.pk',
        },
        'graph_sent': {
            'adj-train-sent': './data/vast_zero/graph_sent/train_sent.graph.adj.pk',
            'adj-dev-sent': './data/vast_zero/graph_sent/dev_sent.graph.adj.pk',
            'adj-test-sent': './data/vast_zero/graph_sent/test_sent.graph.adj.pk',
        },
        'graph_without_LM': {
            'adj-train-without-LM': './data/vast_zero/graph_without_LM/train_without_LM.graph.adj.pk',
            'adj-dev-without-LM': './data/vast_zero/graph_without_LM/dev_without_LM.graph.adj.pk',
            'adj-test-without-LM': './data/vast_zero/graph_without_LM/test_without_LM.graph.adj.pk',
        },
    },
    'vast_few': {
        'statement': {
            'train': './data/vast_few/statement/train.statement.jsonl',
            'dev': './data/vast_few/statement/dev.statement.jsonl',
            'test': './data/vast_few/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/vast_few/grounded/train.grounded.jsonl',
            'dev': './data/vast_few/grounded/dev.grounded.jsonl',
            'test': './data/vast_few/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/vast_few/graph/train.graph.adj.pk',
            'adj-dev': './data/vast_few/graph/dev.graph.adj.pk',
            'adj-test': './data/vast_few/graph/test.graph.adj.pk',
            'adj-train-sent': './data/vast_few/graph/train_sent.graph.adj.pk',
            'adj-dev-sent': './data/vast_few/graph/dev_sent.graph.adj.pk',
            'adj-test-sent': './data/vast_few/graph/test_sent.graph.adj.pk',
        },
        'graph_sent': {
            'adj-train-sent': './data/vast_few/graph_sent/train_sent.graph.adj.pk',
            'adj-dev-sent': './data/vast_few/graph_sent/dev_sent.graph.adj.pk',
            'adj-test-sent': './data/vast_few/graph_sent/test_sent.graph.adj.pk',
        },
        'graph_without_LM': {
            'adj-train-without-LM': './data/vast_few/graph_without_LM/train_without_LM.graph.adj.pk',
            'adj-dev-without-LM': './data/vast_few/graph_without_LM/dev_without_LM.graph.adj.pk',
            'adj-test-without-LM': './data/vast_few/graph_without_LM/test_without_LM.graph.adj.pk',
        },
    },
    'vast_all': {
        'statement': {
            'train': './data/vast_all/statement/train.statement.jsonl',
            'dev': './data/vast_all/statement/dev.statement.jsonl',
            'test': './data/vast_all/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/vast_all/grounded/train.grounded.jsonl',
            'dev': './data/vast_all/grounded/dev.grounded.jsonl',
            'test': './data/vast_all/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/vast_all/graph/train.graph.adj.pk',
            'adj-dev': './data/vast_all/graph/dev.graph.adj.pk',
            'adj-test': './data/vast_all/graph/test.graph.adj.pk',
            'adj-train-sent': './data/vast_all/graph/train_sent.graph.adj.pk',
            'adj-dev-sent': './data/vast_all/graph/dev_sent.graph.adj.pk',
            'adj-test-sent': './data/vast_all/graph/test_sent.graph.adj.pk',
        },
        'graph_sent': {
            'adj-train-sent': './data/vast_all/graph_sent/train_sent.graph.adj.pk',
            'adj-dev-sent': './data/vast_all/graph_sent/dev_sent.graph.adj.pk',
            'adj-test-sent': './data/vast_all/graph_sent/test_sent.graph.adj.pk',
        },
        'graph_without_LM': {
            'adj-train-without-LM': './data/vast_all/graph_without_LM/train_without_LM.graph.adj.pk',
            'adj-dev-without-LM': './data/vast_all/graph_without_LM/dev_without_LM.graph.adj.pk',
            'adj-test-without-LM': './data/vast_all/graph_without_LM/test_without_LM.graph.adj.pk',
        }
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['vast_zero'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'vast_zero': [
            # {'func': convert_vast, 'args': (input_paths['vast_zero']['train'], output_paths['vast_zero']['statement']['train'])},
            # {'func': convert_vast, 'args': (input_paths['vast_zero']['dev'], output_paths['vast_zero']['statement']['dev'])},
            # {'func': convert_vast, 'args': (input_paths['vast_zero']['test'], output_paths['vast_zero']['statement']['test'])},
            # {'func': ground, 'args': (output_paths['vast_zero']['statement']['train'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_zero']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_zero']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_zero']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_zero']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_zero']['grounded']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_zero']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_sent']['adj-train-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_zero']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_sent']['adj-dev-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_zero']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_sent']['adj-test-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_zero']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_without_LM']['adj-train-without-LM'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_zero']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_without_LM']['adj-dev-without-LM'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_zero']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph_without_LM']['adj-test-without-LM'], args.nprocs)},
        ],
        'vast_few': [
            # {'func': convert_vast, 'args': (input_paths['vast_few']['train'], output_paths['vast_few']['statement']['train'])},
            # {'func': convert_vast, 'args': (input_paths['vast_few']['dev'], output_paths['vast_few']['statement']['dev'])},
            # {'func': convert_vast, 'args': (input_paths['vast_few']['test'], output_paths['vast_few']['statement']['test'])},
            # # {'func': ground, 'args': (output_paths['vast_few']['statement']['train'], output_paths['cpnet']['vocab'],
            # #                           output_paths['cpnet']['patterns'], output_paths['vast_few']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_few']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_few']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_few']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_few']['grounded']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_few']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_few']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_few']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_few']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_sent']['adj-train-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_few']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_sent']['adj-dev-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_few']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_sent']['adj-test-sent'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_few']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_without_LM']['adj-train-without-LM'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_few']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_without_LM']['adj-dev-without-LM'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_few']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_few']['graph_without_LM']['adj-test-without-LM'], args.nprocs)},
        ],
        'vast_all': [
            # {'func': convert_vast, 'args': (input_paths['vast_all']['train'], output_paths['vast_all']['statement']['train'])},
            # {'func': convert_vast, 'args': (input_paths['vast_all']['dev'], output_paths['vast_all']['statement']['dev'])},
            # {'func': convert_vast, 'args': (input_paths['vast_all']['test'], output_paths['vast_all']['statement']['test'])},
            # # {'func': ground, 'args': (output_paths['vast_all']['statement']['train'], output_paths['cpnet']['vocab'],
            # #                           output_paths['cpnet']['patterns'], output_paths['vast_all']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_all']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_all']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['vast_all']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['vast_all']['grounded']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_all']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_all']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_all']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_all']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_sent']['adj-train-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_all']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_sent']['adj-dev-sent'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_SentBert, 'args': (output_paths['vast_all']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_sent']['adj-test-sent'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_all']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_without_LM']['adj-train-without-LM'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_all']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_without_LM']['adj-dev-without-LM'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['vast_all']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_all']['graph_without_LM']['adj-test-without-LM'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
