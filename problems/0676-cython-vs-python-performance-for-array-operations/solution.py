import math

def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    
    Args:
        n: Array size (number of elements)
        operations: List of dicts with performance parameters for each operation
    
    Returns:
        dict with per-operation metrics, pipeline totals, recommendation,
        and crossover sizes.
    """
    operationss = []
    cum_python=0
    cum_cython=0
    cum_numpy=0
    crossover_sizes={}
    for operation in operations:
        s = {}
        s.setdefault('name', operation['name'])
        python_us = (operation['python_overhead_ns']+n*operation['python_per_elem_ns'])/1000
        cython_us =  (operation['cython_overhead_ns']+n*operation['cython_per_elem_ns'])/1000
        numpy_us =  (operation['numpy_overhead_ns']+n*operation['numpy_per_elem_ns'])/1000
        if operation['cython_per_elem_ns']<=operation['numpy_per_elem_ns']:
            crossover_sizes.setdefault(operation['name'], -1)
        else:
            size = (operation['numpy_overhead_ns']-operation['cython_overhead_ns'])/(operation['cython_per_elem_ns']-operation['numpy_per_elem_ns'])
            crossover_sizes.setdefault(operation['name'],max(0, math.ceil(size)))
        cum_python+=python_us
        cum_numpy+=numpy_us
        cum_cython+=cython_us
        cython_speedup = python_us/cython_us
        numpy_speedup = python_us/numpy_us
        best = 'cython'
        best_vl = cython_us
        if numpy_us<best_vl:
            best = 'numpy'
            best_vl = numpy_us
        if python_us<best_vl:
            best = 'python'
            best_vl = python_us 
        s.setdefault('python_us', round(python_us, 2))
        s.setdefault('cython_us', round(cython_us,2))
        s.setdefault('numpy_us', round(numpy_us,2))
        s.setdefault('cython_speedup', round(cython_speedup,2))
        s.setdefault('numpy_speedup', round(numpy_speedup,2))
        s.setdefault('best', best)
        operationss.append(s)
    recommended = 'cython'
    recomended_vl = cum_cython
    if cum_numpy<recomended_vl:
        recommended = 'numpy'
        recomended_vl=cum_numpy
    if cum_python<recomended_vl:
        recommended = 'python'
        recomended_vl=cum_python
    return {'operations': operationss, 'pipeline_python_us': round(cum_python,2), 'pipeline_cython_us': round(cum_cython,2), 'pipeline_numpy_us': round(cum_numpy,2), 'recommended': recommended, 'crossover_sizes': crossover_sizes}


