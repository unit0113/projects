import heapq


class Job:
    def __init__(self, duration, weight, name) -> None:
        self.duration = duration
        self.weight = weight
        self.name = name

    @property
    def ratio(self):
        return -(self.weight / self.duration)       # Negative sign is pythonic way to implement max heap with heapq

    def __lt__(self, other):
        return self.ratio < other.ratio

    def __eq__(self, other):
        return self.ratio == other.ratio

    def __repr__(self) -> str:
        return f"Job: {self.name}, Importance: {self.weight}, Duration: {self.duration}"

class Scheduler:
    def __init__(self, jobs: list[Job] = None) -> None:
        self.jobs = []
        if jobs:
            for job in jobs:
                heapq.heappush(self.jobs, job)

    def add_job(self, job):
        heapq.heappush(self.jobs, job)

    def schedule(self):
        total_time = 0
        while self.jobs:
            job = heapq.heappop(self.jobs)
            total_time += job.duration
            print(f"{job} Completed at {total_time}")


schd = Scheduler()
schd.add_job(Job(1, 1, "PowerPoint"))
schd.add_job(Job(3, 1, "Boring Meeting"))
schd.add_job(Job(1, 2, "Check Email"))
schd.add_job(Job(2, 4, "Program"))
schd.add_job(Job(3, 5, "Learn New Stuff"))
schd.schedule()