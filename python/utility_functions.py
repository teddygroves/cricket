from hashlib import md5
import numpy as np
import os
import pandas as pd
import pickle
import pystan


def get_flat_columns(df, sep='_'):
    """Takes a multi-level column index and turns it into a one-level
    index where the level names are connected by a specified separator
    (defults to '_').  This can be useful after `pd.unstack(...)` or
    after a aggregation with multiple functions.

    """
    if isinstance(df.columns, pd.MultiIndex):
        return [sep.join(list(map(str, col))).strip(sep) for col in df.columns.values]
    else:
        return df.columns.values


def flatten_columns(df, sep='_'):
    """A function that takes in a dataframe with possibly multi-level
    columns and returns the same dataframe with flattened columns.

    """
    new_df = df.copy()
    new_df.columns = get_flat_columns(df, sep)
    return new_df


def z_score(value, mean=None, std_dev=None):
    if mean is None:
        mean = np.mean(value)
    if std_dev is None:
        std_dev = np.std(value)

    return (value - mean)/std_dev


def stanify_series(input_series: pd.Series, sort=False):
    """replace the values of a pandas series with 1-indexed codes

    :param input_series: pandas series

    :param sort: do you want the order of the series to be preserved?

    :return: series where the values of input_series are translated
    into 1-indexed integer codes

    """
    return pd.Series(
        input_series
        .fillna('value_missing')
        .factorize(sort=sort)[0],
        index=input_series.index
    ).add(1)


def get_stan_path(file_path):
    """Get a path to a .stan file in the stan directory 

    :param file_path: a path to the required .stan file, starting from
    the stan directory 

    :return: a path that works with the `file=` argument of Stan
    functions

    """
    stan_path = os.path.dirname(
        os.path.realpath(__file__)) + "/../stan/"
    return stan_path + file_path


def get_sample_file(file_name):
    """Get the address of file to dump samples in

    Example usage:

      file = get_sample_file('my_model_samples')
      fit = my_model.sampling(data=my_data, sample_file=file)

    This will result in some csv files (one per chain) being written
    to the directory `modelling/data/csv/samples`, each with the name
    `my_model_samples_[CHAIN NUMBER].csv`

    """
    return (
        os.path.dirname(os.path.realpath(__file__)) +
        "/../data/samples/{}.csv".format(file_name)
    )


def read_samples(sample_file, n_chains, first_non_warmup):
    """Get a dataframe from csvs of dumped samples

    :param sample_file: string of address for samples, e.g. output of
    get_sample_file above
    :param n_chains: number of sample csvs to look for
    :param first_non_warmup: position of the first non-warmup sample

    :return: a samples dataframe that can be used to get fit
    diagnostics and parameter samples using the functions
    get_diagnostic_df and get_parameter_samples below

    """

    sample_list = [
        pd.read_csv(sample_file[:-4] + "_{0}.csv".format(chain), comment='#')
        .iloc[first_non_warmup:]
        .reset_index()
        .rename(columns={'index': 'iteration_number'})
        for chain in range(n_chains)
    ]

    return pd.concat(sample_list, ignore_index=True)


def get_diagnostic_df(samples, first_non_warmup=0):
    """Get diagnostic columns from a samples dataframe"""

    diagnostic_columns = [i for i in samples.columns if i[-2:] == "__"]
    return samples[diagnostic_columns]


def get_parameter_samples(parameter_name, samples):
    """get a nicely indexed dataframe of a parameter's samples from a
    samples dataframe"""

    def try_to_make_int(thing):
        try:
            return int(thing)
        except ValueError:
            return thing

    old_columns = [i for i in samples.columns if i.split('.')[0] == parameter_name]

    new_columns = pd.MultiIndex.from_tuples([
        tuple(map(try_to_make_int, i.split('.')))
        for i in old_columns
    ])

    wanted_samples = samples[old_columns].copy()
    wanted_samples.columns = new_columns

    return wanted_samples.sort_index(axis=1)[parameter_name]


def compile_stan_model_with_cache(model_file, model_name=None):
    """Avoid recompiling a Stan model if possible"""
    with open(model_file, 'r') as stan_program:
        model_code = stan_program.read()
    path_to_dump_area = os.path.dirname(
        os.path.realpath(__file__)) + "/../data/pickled_stan_models/"
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        dump_string = "{}cached_model-{}.pkl".format(
            path_to_dump_area, code_hash
        )
    else:
        dump_string = "{}cached_model-{}-{}.pkl".format(
            path_to_dump_area, model_name, code_hash
        )
    try:
        model = pickle.load(open(dump_string, 'rb'))
    except FileNotFoundError:
        model = pystan.StanModel(model_code=model_code)
        with open(dump_string, 'wb') as f:
            pickle.dump(model, f)
    return model
