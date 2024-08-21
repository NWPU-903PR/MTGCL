import os, argparse
from scipy.stats import ttest_ind

from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef
def parse_args():
    parser = argparse.ArgumentParser(description='Train model with cross-validation and save model to file')
    parser.add_argument('-cancer', '--cancer', dest='cancer', type=str, default='CPDB')
    parser.add_argument('-model', '--model', dest='model', type=str, default='MTGCL')
    parser.add_argument('-device', '--device', dest='device', type=str, default='cuda:0')
    parser.add_argument('-sd', '--seed', dest='seed', type=int, default=12345)
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=1900,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=0.001,
                        type=float
                        )
    parser.add_argument('-wd', '--weight_decay', help='Weight Decay',
                        dest='decay',
                        default=0,
                        type=float
                        )
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer. Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        default=[300, 100])

    parser.add_argument('-dropout', '--dropout', help='Dropout Percentage',
                        dest='dropout',
                        default=0.5,
                        type=float
                        )

    parser.add_argument('-dropout_edge', '--dropout_edge', help='Dropout Edge Percentage',
                        dest='dropout_edge',
                        default=0.5,
                        type=float
                        )

    parser.add_argument('-pe1', '--pe1', help='drop_edge_rate_1',
                        dest='pe1',
                        default=0.1,
                        type=float
                        )
    parser.add_argument('-pe2', '--pe2', help='drop_edge_rate_2',
                        dest='pe2',
                        default=0.3,
                        type=float
                        )
    parser.add_argument('-pf1', '--pf1', help='drop_feature_rate_1',
                        dest='pf1',
                        default=0.1,
                        type=float
                        )
    parser.add_argument('-pf2', '--pf2', help='drop_feature_rate_2',
                        dest='pf2',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-t', '--tau', help='tau',
                        dest='tau',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-lamuta', '--λ', help='lamuta',
                        dest='λ',
                        default=0.7,
                        type=float
                        )
    parser.add_argument('-cv', '--cv_runs', help='number of folds for cross-validation',
                    dest='cv_runs',
                    default=5,
                    type=int
                    )
    parser.add_argument('-num', '--cv_runs_num', help='number of iterations for cross-validation',
                        dest='num',
                        default=10,
                        type=int
                        )
    args = parser.parse_args()
    return args

def create_model_dir(root_dir,a):
    #date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = os.path.join(root_dir, 'model' + '_' + str(a) + '/')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    return model_path
def create_train_dir(root_dir,a):
    if not os.path.isdir(root_dir):  # in case training root doesn't exist
        os.mkdir(root_dir)
        print("Created Training Subdir")
    #date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = os.path.join(root_dir, 'train' + '_' + str(a))
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    return model_path
def create_val_dir(root_dir,a):
    if not os.path.isdir(root_dir):  # in case training root doesn't exist
        os.mkdir(root_dir)
        print("Created Training Subdir")
    #date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = os.path.join(root_dir, 'val' + '_' + str(a))
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    return model_path


def write_hyper_params(args, file_name):
    """Write hyper parameters to disk.

    Writes a set of hyper parameters of the model to disk.
    See `load_hyper_params` for information on how to load
    the hyper parameters.

    Parameters:
    ----------
    args:               The parameters to save as dictionary
    input_file:         The input data hdf5 container. Only
                        present for legacy reasons
    file_name:          The file name to write the data to.
                        Should be 'hyper_params.txt' in order
                        for the load function to work properly.
    """
    with open(file_name, 'w') as f:
        for arg in args:
            f.write('{}\t{}\n'.format(arg, args[arg]))
        #f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))