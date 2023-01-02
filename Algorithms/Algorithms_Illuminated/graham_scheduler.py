def graham(jobs: list, num_machines: int, *, reverse=False) -> list:
    assignments = [[] for _ in range(num_machines)]
    loads = [0] * num_machines
    
    if reverse:
        jobs.sort(reverse=True)
    else:
        jobs.sort()
    
    for job in jobs:
        min_machine = loads.index(min(loads))
        assignments[min_machine].append(job)
        loads[min_machine] += job

    return assignments


jobs = [1] * 20 + [5]
num_machines = 5
print(graham(jobs, num_machines))
print(graham(jobs, num_machines, reverse=True))