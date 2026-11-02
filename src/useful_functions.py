import xarray as xr
import numpy as np


def data_between(y, q_low, X=None, q_high=None):
    # get the datapoints within a quantile range, measured from cumculative sum on y
    # if q_high is None, then select only the y closest to q_low
    sum_array = np.sort(np.sum(y, axis=-1))
    N = len(sum_array)

    # collect indexes for low and high sum boundaries
    # round of i_low, i_high is the 'nearest' method, effectively qval=0.7666 is 0.76 for example
    # interpolation is not feasiable in this setting
    i_low = min(int(q_low * N), N - 1)
    if q_high is not None:
        i_high = min(int(q_high * N), N - 1)

    y_low_sum = sum_array[i_low]
    if q_high is None:
        y_high_sum = y_low_sum
    else:
        y_high_sum = sum_array[i_high]

    y_low_sum = np.round(y_low_sum)
    y_high_sum = np.round(y_high_sum)

    mask = [True if y_low_sum <= np.round(np.sum(yi)) <= y_high_sum else False for yi in y]
    if X is not None:
        return X[mask], y[mask]
    else:
        return y[mask]


# mean of list of xarrays, avoids conversion to numpy array
def xarray_mean(preds, new_names={'dim_0': 'lon', 'dim_1':'lat'}):
    xarr = xr.DataArray(np.sum(preds, axis=0))
    xarr *= 1 / len(preds)
    #xarr = xarr.rename(new_names)

    return xarr


# evaluate then average over runs
def evaluate_models(
                        models,
                        inputs,
                        take_mean=True,
                    ):
    model_out = []
    # if not None, take mean of prediction
    for n, model in enumerate(models):
        pred = model(inputs)
        model_out.append(pred)

    if take_mean:
        return xarray_mean(model_out)
    else:
        return model_out
