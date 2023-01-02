import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment,convert_vast
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

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
            'adj-train_1':  './data/vast_zero/graph/train_1.graph.adj.pk',
            'adj-train_2':  './data/vast_zero/graph/train_2.graph.adj.pk',
            'adj-train_3':  './data/vast_zero/graph/train_3.graph.adj.pk',
            'adj-train_4':  './data/vast_zero/graph/train_4.graph.adj.pk',
            'adj-train_5':  './data/vast_zero/graph/train_5.graph.adj.pk',
            'adj-train_6':  './data/vast_zero/graph/train_6.graph.adj.pk',
            'adj-train_7':  './data/vast_zero/graph/train_7.graph.adj.pk',
            'adj-train_8':  './data/vast_zero/graph/train_8.graph.adj.pk',
            'adj-train_9':  './data/vast_zero/graph/train_9.graph.adj.pk',
            'adj-train_10': './data/vast_zero/graph/train_10.graph.adj.pk',
            'adj-train_11': './data/vast_zero/graph/train_11.graph.adj.pk',
            'adj-train_12': './data/vast_zero/graph/train_12.graph.adj.pk',
            'adj-train_13': './data/vast_zero/graph/train_13.graph.adj.pk',
            'adj-train_14': './data/vast_zero/graph/train_14.graph.adj.pk',
        },
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': './data/obqa/statement/train-fact.statement.jsonl',
            'dev': './data/obqa/statement/dev-fact.statement.jsonl',
            'test': './data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test-fact.jsonl',
        },
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
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['train_10'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-train_10'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['vast_zero']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['vast_zero']['graph']['adj-test'], args.nprocs)},
        ],

        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],
        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
