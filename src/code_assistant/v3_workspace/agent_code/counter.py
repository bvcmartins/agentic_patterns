class BoundedCounter:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.counter = dict()
        self.order = list()

    def add(self, key: str):
        if key in self.counter:
            self.counter[key] += 1
            self.order.remove(key)
            self.order.append(key)
        else:
            if len(self.order) >= self.capacity:
                evict_key = self.order.pop(0)
                del self.counter[evict_key]
            self.counter[key] = 1
            self.order.append(key)

    def top(self) -> str:
        max_count = -1
        max_key = None
        for key, count in self.counter.items():
            if count > max_count:
                max_count = count
                max_key = key
            elif count == max_count:
                if max_key is None:
                    max_key = key
        return max_key

    def keys(self) -> list:
        return self.order