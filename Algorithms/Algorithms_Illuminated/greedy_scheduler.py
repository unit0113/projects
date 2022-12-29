import heapq


class Scheduler:
    def __init__(self, jobs: list[tuple[int, str]] = None) -> None:
        self.jobs = []
        if jobs:
            for job in jobs:
                heapq.heappush(self.jobs, job)

    def add_job(self, job):
        heapq.heappush(self.jobs, job)

    def schedule(self):
        total_time = 0
        while self.jobs:
            duration, job = heapq.heappop(self.jobs)
            total_time += duration
            print(f"{job}: {duration}, Completed at {total_time}")


schd = Scheduler()
schd.add_job((1, "PowerPoint"))
schd.add_job((3, "Boring Meeting"))
schd.add_job((1, "Check Email"))
schd.add_job((2, "Program"))
schd.add_job((3, "Learn New Stuff"))
schd.schedule()