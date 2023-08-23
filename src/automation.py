import os
from experiments.vary_fault_tolerance import vary_tolerant
from run_parallel.shutdown import shutdown
from run_parallel.connection import connect
from run_parallel.tasks import runCommand
from run_parallel.common import LOG_DIR, network
from experiments.strong_attack import strong_attack
from experiments.baseline import baseline
from experiments.single_adversary import single_adversary
from experiments.vary_adversary_count import vary
from experiments.vary_selection import vary_selection
from experiments.vary_selection_local import vary_selection_local
from experiments.vary_pagerank import vary_page_rank
from experiments.vary_nodecount import vary_node_count
from experiments.vary_graph import vary_graph
from experiments.vary_dataset_fashion import vary_dataset_fashion
from run_parallel.common import command
from experiments.vary_clean import clean_graph_type
from experiments.vary_nodecount_clean import node_count_clean
from experiments.baseline_fault_tolerant import vary_baseline_tolerant
from experiments.baseline_clipping import vary_baseline_clipping
from experiments.clipping_strongest import clipping_strongest
from experiments.vary_agg import vary_agg
from experiments.vary_agg_clip import vary_agg_clip
from experiments.vary_clipping_norm import vary_clipping_norm
from experiments.local_norm import local_norm
from experiments.noniid_attack import noniid_attack
from experiments.noniid_attack_defense import noniid_attack_defense
from experiments.noniid_attack_local import noniid_attack_local
from experiments.noniid_local_attack_defense import noniid_local_attack_defense
from experiments.vary_selection_local_defense import vary_selection_local_defense
from experiments.vary_selection_local_more import vary_selection_local_more
from experiments.noniid_attack_more import noniid_attack_more
import time

experiment_map = {
    'Baseline':baseline,
    'StrongAttack':strong_attack,
    'SingleAdversary':single_adversary,
    'Vary':vary,
    'Selection':vary_selection,
    'Pagerank' :vary_page_rank,
    'NodeCount':vary_node_count,
    'Graph':vary_graph,
    'VaryFashion':vary_dataset_fashion,
    'VaryClean':clean_graph_type,
    'VaryNodeCountClean':node_count_clean,
    'Tolerant':vary_tolerant,
    'BaselineTolerant':vary_baseline_tolerant,
    'BaselineClipping':vary_baseline_clipping,
    'ClippingStrongest':clipping_strongest,
    'VaryAgg':vary_agg,
    'VaryAggClip':vary_agg_clip,
    'VaryClippingNorm':vary_clipping_norm,
    'LocalNorm':local_norm,
    'VarySelectionLocal':vary_selection_local,
    'Non-iidAttack':noniid_attack,
    'Non-iidAttackDefense':noniid_attack_defense,
    'Non-iidAttackLocal':noniid_attack_local,
    'Non-iidLocalAttackDefense':noniid_local_attack_defense,
    'VarySelectionLocalDefense':vary_selection_local_defense,
    'VarySelectionLocal_more':vary_selection_local_more,
    'Non-iidAttackMore': noniid_attack_more

}

def runDocker():
    os.system(f'docker run -d  -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 -v {network["workDir"]}/run_parallel/myrabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq:3.9-management')
    os.system('docker run  -d -it --rm --name flower   -p 5555:5555 mher/flower:0.9.5 --broker="amqp://guest:guest@`hostname`:5672" --broker_api="http://guest:guest@`hostname`:15672/api/vhost" --address=0.0.0.0')
    connect()

    
def stopDocker():
    os.system('docker stop flower rabbitmq')
    shutdown()



def setup_argparse():
    import argparse 
    parser = argparse.ArgumentParser('p2pml-cli')
    subparsers = parser.add_subparsers(help='Choose one of the following',dest='command')


    parser_run= subparsers.add_parser('run', help='Automaticly run experiments')
    parser_run.add_argument('start', metavar='s', type=int,help= 'starting experiment inclusive')
    parser_run.add_argument('end', metavar='e', type=int, help= 'ending experiment exclusive')
    
    parser_run= subparsers.add_parser('docker', help='Start/Stop Docker instance')
    parser_run.add_argument('docker', choices=['start','stop'], help= 'start/stop docker instance')

    parser_run= subparsers.add_parser('experiment', help='Create experiments files')
    parser_run.add_argument('experiment', choices=list(experiment_map.keys()) + ['all'], help= 'choose experiment')

    return parser

def main():
    
    parser = setup_argparse()
    args = parser.parse_args()

    if args.command == 'experiment':
        if args.experiment == 'all':
            experimentCount = 0
            for experiment in experiment_map.values():
                experimentCount += experiment()
        else:
            experimentCount = experiment_map[args.experiment]()

    elif args.command == 'run':
        for i in range(args.start, args.end):
            logPath = os.path.join(LOG_DIR, f'experiment-{i}.txt')
            job = f'{command} {i} >{logPath} 2>{logPath} '
            print(job)
            runCommand.delay(job)
            time.sleep(0.1)  

    elif args.command == 'docker':
        if args.docker == 'start':
            runDocker()
        elif args.docker == 'stop':
            stopDocker()