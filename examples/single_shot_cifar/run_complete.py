"""
Run the complete experiment and plot the results

The experiment runs in three phases
1. The training in parallel
2. The evaluating in parallel on the train and the test sets
3. Plot the results
"""
import subprocess
import logging

logging.basicConfig(format='Running expeiments -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)

# start the training and wait for it
logging.info('Running the training...')
with open('log_train_bp.log', 'w') as out_file:
    train_bp = subprocess.Popen(['python', '-m',
                                 'pseudo_backprop.experiments.train_mnist',
                                 '--params', 'params_vbp.json'],
                                stdout=out_file, stderr=out_file, shell=False)
with open('log_train_fa.log', 'w') as out_file:
    train_fa = subprocess.Popen(['python', '-m',
                                 'pseudo_backprop.experiments.train_mnist',
                                 '--params', 'params_fa.json'],
                                stdout=out_file, stderr=out_file, shell=False)
with open('log_train_pseudo.log', 'w') as out_file:
    train_pbp = subprocess.Popen(['python', '-m',
                                  'pseudo_backprop.experiments.train_mnist',
                                  '--params', 'params_pseudo_backprop.json'],
                                  stdout=out_file, stderr=out_file,
                                  shell=False)
# wait for the training
wait_for = [p.wait() for p in (train_bp, train_fa, train_pbp)]
logging.info('Training has finished')

# start the evaluation and wait for it to finish
logging.info('Running the evaluation...')
with open('log_eval_test_bp.log', 'w') as out_file:
    eval_bp_test = subprocess.Popen(['python', '-m',
                                     'pseudo_backprop.experiments.test_mnist',
                                     '--params', 'params_vbp.json',
                                     '--dataset', 'test'],
                                    stdout=out_file, stderr=out_file,
                                    shell=False)
with open('log_eval_train_bp.log', 'w') as out_file:
    eval_bp_train = subprocess.Popen(['python',  '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                      '--params', 'params_vbp.json',
                                      '--dataset', 'train'],
                                     stdout=out_file, stderr=out_file,
                                     shell=False)
with open('log_eval_test_fa.log', 'w') as out_file:
    eval_fa_test = subprocess.Popen(['python', '-m',
                                     'pseudo_backprop.experiments.test_mnist',
                                     '--params', 'params_fa.json',
                                     '--dataset', 'test'],
                                    stdout=out_file, stderr=out_file,
                                    shell=False)
with open('log_eval_train_fa.log', 'w') as out_file:
    eval_fa_train = subprocess.Popen(['python',  '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                      '--params', 'params_fa.json',
                                      '--dataset', 'train'],
                                     stdout=out_file, stderr=out_file,
                                     shell=False)
with open('log_eval_test_pseudo_backprop.log', 'w') as out_file:
    eval_pbp_test = subprocess.Popen(['python',  '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                     '--params', 'params_pseudo_backprop.json',
                                     '--dataset', 'test'],
                                    stdout=out_file, stderr=out_file,
                                    shell=False)
with open('log_eval_train_pseudo_backprop.log', 'w') as out_file:
    eval_pbp_train = subprocess.Popen(['python', '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                      '--params',
                                      'params_pseudo_backprop.json',
                                      '--dataset', 'train'],
                                     stdout=out_file, stderr=out_file,
                                     shell=False)
wait_for = [p.wait() for p in (eval_bp_test, eval_bp_train,
                               eval_fa_test, eval_fa_train,
                               eval_pbp_test, eval_pbp_train)]
logging.info('Evaluation has finished')

logging.info('Start the plotting...')
plot_call = ['python',  '-m', 'pseudo_backprop.experiments.plot_mnist_results',
             '--params_vbp', 'params_vbp.json',
             '--params_fa', 'params_fa.json',
             '--params_pseudo', 'params_pseudo_backprop.json']
subprocess.Popen(plot_call, shell=False)
logging.info('The plotting has finished')
