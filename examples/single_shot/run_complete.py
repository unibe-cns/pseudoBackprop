"""
Run the complete experiment and plot the results

The experiment runs in three phases
1. The training in parallel
2. The evaluating in parallel on the train and the test sets
3. Plot the results
"""
import subprocess
import logging
import shutil

logging.basicConfig(format='Running experiments -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)

SKIP_TRAIN = False

# start the training and wait for it
logging.info('Running the training...')
with open('log_train_bp.log', 'w') as out_file:
    train_bp = subprocess.Popen(['python3', '-m',
                                 'pseudo_backprop.experiments.train_mnist',
                                 '--params', 'params_vbp.json'],
                                stdout=out_file, stderr=out_file, shell=False)
with open('log_train_fa.log', 'w') as out_file:
    train_fa = subprocess.Popen(['python3', '-m',
                                 'pseudo_backprop.experiments.train_mnist',
                                 '--params', 'params_fa.json'],
                                stdout=out_file, stderr=out_file, shell=False)
with open('log_train_pseudo.log', 'w') as out_file:
    train_pbp = subprocess.Popen(['python3', '-m',
                                  'pseudo_backprop.experiments.train_mnist',
                                  '--params', 'params_pseudo_backprop.json'],
                                 stdout=out_file, stderr=out_file,
                                 shell=False)
with open('log_train_gen_pseudo.log', 'w') as out_file:
    train_gen = subprocess.Popen(['python3', '-m',
                                  'pseudo_backprop.experiments.train_mnist',
                                  '--params', 'params_gen_pseudo.json'],
                                 stdout=out_file, stderr=out_file,
                                 shell=False)

# wait for the training
wait_for = [p.wait() for p in (train_bp, train_fa, train_pbp, train_gen)]
logging.info('Training has finished')


# run for plotting the acitvities
logging.info('Running the measurement of the activities')
exp_params = ['params_fa.json', 'params_gen_pseudo.json',
              'params_pseudo_backprop.json', 'params_vbp.json']
processes_arr = []
for exp in exp_params:
    with open(f'log_activities_{exp[:-5]}.log', 'w') as out_file:
        processes_arr.append(subprocess.Popen(
            ['python3', '-m',
             'pseudo_backprop.experiments.measure_activities',
             '--params', exp],
            stdout=out_file, stderr=out_file,
            shell=False))

wait_for = [p.wait() for p in processes_arr]
logging.info('Measuring the activities has finished')

# start the evaluation and wait for it to finish
logging.info('Running the evaluation...')
with open('log_eval_test_bp.log', 'w') as out_file:
    eval_bp_test = subprocess.Popen(['python3', '-m',
                                     'pseudo_backprop.experiments.test_mnist',
                                     '--params', 'params_vbp.json',
                                     '--dataset', 'test'],
                                    stdout=out_file, stderr=out_file,
                                    shell=False)
if not SKIP_TRAIN:
  with open('log_eval_train_bp.log', 'w') as out_file:
      eval_bp_train = subprocess.Popen(['python3',  '-m',
                                        'pseudo_backprop.experiments.test_mnist',
                                        '--params', 'params_vbp.json',
                                        '--dataset', 'train'],
                                       stdout=out_file, stderr=out_file,
                                       shell=False)
with open('log_eval_test_fa.log', 'w') as out_file:
    eval_fa_test = subprocess.Popen(['python3', '-m',
                                     'pseudo_backprop.experiments.test_mnist',
                                     '--params', 'params_fa.json',
                                     '--dataset', 'test'],
                                    stdout=out_file, stderr=out_file,
                                    shell=False)
if not SKIP_TRAIN:
  with open('log_eval_train_fa.log', 'w') as out_file:
      eval_fa_train = subprocess.Popen(['python3',  '-m',
                                        'pseudo_backprop.experiments.test_mnist',
                                        '--params', 'params_fa.json',
                                        '--dataset', 'train'],
                                       stdout=out_file, stderr=out_file,
                                       shell=False)
with open('log_eval_test_pseudo_backprop.log', 'w') as out_file:
    eval_pbp_test = subprocess.Popen(['python3',  '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                      '--params', 'params_pseudo_backprop.json',
                                      '--dataset', 'test'],
                                     stdout=out_file, stderr=out_file,
                                     shell=False)
if not SKIP_TRAIN:
  with open('log_eval_train_pseudo_backprop.log', 'w') as out_file:
      eval_pbp_train = subprocess.Popen(['python3', '-m',
                                         'pseudo_backprop.experiments.test_mnist',
                                         '--params',
                                         'params_pseudo_backprop.json',
                                         '--dataset', 'train'],
                                        stdout=out_file, stderr=out_file,
                                        shell=False)
with open('log_eval_test_gen_pseudo.log', 'w') as out_file:
    eval_gen_test = subprocess.Popen(['python3',  '-m',
                                      'pseudo_backprop.experiments.test_mnist',
                                      '--params', 'params_gen_pseudo.json',
                                      '--dataset', 'test'],
                                     stdout=out_file, stderr=out_file,
                                     shell=False)
if not SKIP_TRAIN:
  with open('log_eval_train_gen_pseudo.log', 'w') as out_file:
      eval_gen_train = subprocess.Popen(['python3', '-m',
                                         'pseudo_backprop.experiments.test_mnist',
                                         '--params',
                                         'params_gen_pseudo.json',
                                         '--dataset', 'train'],
                                        stdout=out_file, stderr=out_file,
                                        shell=False)

if SKIP_TRAIN:
  wait_for = [p.wait() for p in (eval_bp_test,
                                 eval_fa_test,
                                 eval_pbp_test,
                                 eval_gen_test)]
  shutil.copyfile('model_bp/results_test.csv', 'model_bp/results_train.csv')
  shutil.copyfile('model_fa/results_test.csv', 'model_fa/results_train.csv')
  shutil.copyfile('model_gen_pseudo/results_test.csv',
                  'model_gen_pseudo/results_train.csv')
  shutil.copyfile('model_pseudo/results_test.csv',
                  'model_pseudo/results_train.csv')
else:
  wait_for = [p.wait() for p in (eval_bp_test, eval_bp_train,
                                 eval_fa_test, eval_fa_train,
                                 eval_pbp_test, eval_pbp_train,
                                 eval_gen_test, eval_gen_train)]
logging.info('Evaluation has finished')


logging.info('Start the plotting...')
plot_call = ['python3',  '-m', 'pseudo_backprop.experiments.plot_mnist_results',
             '--params_vbp', 'params_vbp.json',
             '--params_fa', 'params_fa.json',
             '--params_pseudo', 'params_pseudo_backprop.json',
             '--params_gen_pseudo', 'params_gen_pseudo.json']
subprocess.Popen(plot_call, shell=False)
logging.info('The plotting has finished')
