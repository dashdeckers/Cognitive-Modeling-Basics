from model import Model
from dmchunk import Chunk


numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six']


def init_model():
    # create a model
    m = Model()

    # instantiate the declarative knowledge of how to count to 6
    for num1, num2 in zip(numbers, numbers[1:]):
        fact = Chunk(
            name=f'cf_{num1}-{num2}',
            slots={'isa': 'count-fact', 'num1': num1, 'num2': num2}
        )
        m.add_encounter(fact)

    return m


def count_from(m, start, end):
    # formulate the task at hand as a 'goal'
    m.goal = Chunk(
        name='goal',
        slots={'isa': 'count-goal', 'start': start, 'end': end, 'current': start}
    )

    while not m.goal.slots['current'] == m.goal.slots['end']:
        # formulate a request for the next number after 'current'
        request = Chunk(
            name='request',
            slots={'isa': 'count-fact', 'num1': m.goal.slots['current']}
        )
        # add the time it takes to create the request
        m.time += 0.05

        # retrieve the chunk from declarative memory
        chunk, latency = m.retrieve(request)
        m.add_encounter(chunk)
        # add the time it takes to retrieve the chunk
        m.time += latency

        # add the time it takes to say a number
        m.time += 0.3
        # print the number that was just said and the time elapsed
        print(m.time)
        print(m.goal.slots['current'])

        # update current so we can look for the next number
        m.goal.slots['current'] = chunk.slots['num2']

    print(m.time)
    print(m.goal.slots['current'])


def add(m, num1, num2):
    # formulate the task at hand as a 'goal'
    m.goal = Chunk(
        name='goal',
        slots={'isa': 'add-goal', 'sum': num1, 'counter': 'zero'}
    )

    while not m.goal.slots['counter'] == num2:
        # formulate a request for the next number after the current sum
        sum_request = Chunk(
            name='request',
            slots={'isa': 'count-fact', 'num1': m.goal.slots['sum']}
        )
        # add the time it takes to create the request
        m.time += 0.05
        # retrieve the chunk from declarative memory
        sum_chunk, sum_latency = m.retrieve(sum_request)
        m.add_encounter(sum_chunk)
        # add the time it takes to retrieve the chunk
        m.time += sum_latency

        # formulate a request for the next number after the current counter
        cnt_request = Chunk(
            name='request',
            slots={'isa': 'count-fact', 'num1': m.goal.slots['counter']}
        )
        # add the time it takes to create the request
        m.time += 0.05
        # retrieve the chunk from declarative memory
        cnt_chunk, cnt_latency = m.retrieve(cnt_request)
        m.add_encounter(cnt_chunk)
        # add the time it takes to retrieve the chunk
        m.time += cnt_latency

        # add the time it takes to say a number
        m.time += 0.3
        # print the number that was just said and the time elapsed
        print(m.time)
        print(m.goal.slots['sum'])

        # update the sum so we know where we are
        m.goal.slots['sum'] = sum_chunk.slots['num2']
        # update the counter so we know when to stop
        m.goal.slots['counter'] = cnt_chunk.slots['num2']

    print(m.time)
    print(m.goal.slots['sum'])


if __name__ == '__main__':
    m = init_model()
    add(m, 'two', 'four')
