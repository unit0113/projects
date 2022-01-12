import numpy as np

def calculate(values):
    if len(values) != 9:
        raise ValueError('List must contain nine numbers.')
    values = np.array(values)
    values = values.reshape((3,3))

    output = {}
    output['mean'] = [np.mean(values, axis=0).tolist(), np.mean(values, axis=1).tolist(), np.mean(values).tolist()]
    output['variance'] = [np.var(values, axis=0).tolist(), np.var(values, axis=1).tolist(), np.var(values).tolist()]
    output['standard deviation'] = [np.std(values, axis=0).tolist(), np.std(values, axis=1).tolist(), np.std(values).tolist()]
    output['max'] = [np.max(values, axis=0).tolist(), np.max(values, axis=1).tolist(), np.max(values).tolist()]
    output['min'] = [np.min(values, axis=0).tolist(), np.min(values, axis=1).tolist(), np.min(values).tolist()]
    output['sum'] = [np.sum(values, axis=0).tolist(), np.sum(values, axis=1).tolist(), np.sum(values).tolist()]

    return output

values = [0,1,2,3,4,5,6,7,8]

print(calculate(values))



